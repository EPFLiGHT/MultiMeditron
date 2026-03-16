"""
Error Overlap Analysis for Translation vs Native Pipelines

Analyzes which questions each pipeline gets wrong:
- Do they fail on the SAME questions or DIFFERENT questions?
- How much error overlap exists?
- What's the potential gain from an ensemble approach?

Processes results from 1000 samples per language (zh, es, fr, ja, ru).

Output: src/multimeditron/translation/experiments/results/base_nllb/experiment_0_error_overlap_analysis.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict


class ErrorOverlapAnalyzer:
    """Analyzes error overlap between translation and native pipelines."""
    
    def __init__(self, results_file: str):
        """Load experiment results."""
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.translation_results = self.data['detailed_results']['translation']
        self.native_results = self.data['detailed_results']['native']
        
        assert len(self.translation_results) == len(self.native_results), \
            "Translation and native results must have same length!"
    
    def analyze_overlap(self) -> Dict:
        """Compute error overlap metrics between pipelines."""
        both_correct = []
        both_wrong = []
        trans_correct_native_wrong = []
        native_correct_trans_wrong = []
        
        by_language = defaultdict(lambda: {
            'both_correct': 0,
            'both_wrong': 0,
            'trans_only_correct': 0,
            'native_only_correct': 0,
            'total': 0
        })
        
        for trans, native in zip(self.translation_results, self.native_results):
            assert trans['question'] == native['question'], "Question mismatch!"
            
            lang = trans['language']
            trans_correct = trans['is_correct']
            native_correct = native['is_correct']
            
            by_language[lang]['total'] += 1
            
            if trans_correct and native_correct:
                both_correct.append({
                    'question': trans['question'],
                    'language': lang,
                    'correct_answer': trans['correct']
                })
                by_language[lang]['both_correct'] += 1
                
            elif not trans_correct and not native_correct:
                both_wrong.append({
                    'question': trans['question'],
                    'language': lang,
                    'correct_answer': trans['correct'],
                    'trans_predicted': trans['predicted'],
                    'native_predicted': native['predicted']
                })
                by_language[lang]['both_wrong'] += 1
                
            elif trans_correct and not native_correct:
                trans_correct_native_wrong.append({
                    'question': trans['question'],
                    'language': lang,
                    'correct_answer': trans['correct'],
                    'native_predicted': native['predicted']
                })
                by_language[lang]['trans_only_correct'] += 1
                
            else:
                native_correct_trans_wrong.append({
                    'question': trans['question'],
                    'language': lang,
                    'correct_answer': trans['correct'],
                    'trans_predicted': trans['predicted']
                })
                by_language[lang]['native_only_correct'] += 1
        
        total = len(self.translation_results)
        
        overlap = {
            'total_samples': total,
            'both_correct': len(both_correct),
            'both_wrong': len(both_wrong),
            'trans_only_correct': len(trans_correct_native_wrong),
            'native_only_correct': len(native_correct_trans_wrong),
            'both_correct_pct': len(both_correct) / total * 100,
            'both_wrong_pct': len(both_wrong) / total * 100,
            'trans_only_correct_pct': len(trans_correct_native_wrong) / total * 100,
            'native_only_correct_pct': len(native_correct_trans_wrong) / total * 100,
        }
        
        total_trans_errors = len(both_wrong) + len(native_correct_trans_wrong)
        total_native_errors = len(both_wrong) + len(trans_correct_native_wrong)
        
        overlap['translation_errors'] = total_trans_errors
        overlap['native_errors'] = total_native_errors
        
        if total_trans_errors > 0:
            overlap['error_overlap_pct'] = len(both_wrong) / total_trans_errors * 100
        else:
            overlap['error_overlap_pct'] = 0
        
        ensemble_correct = len(both_correct) + len(trans_correct_native_wrong) + len(native_correct_trans_wrong)
        overlap['ensemble_potential_accuracy'] = ensemble_correct / total * 100
        overlap['ensemble_potential_gain'] = overlap['ensemble_potential_accuracy'] - max(
            self.data['statistics']['translation']['accuracy'],
            self.data['statistics']['native']['accuracy']
        )
        
        overlap['examples'] = {
            'both_correct': both_correct[:5],
            'both_wrong': both_wrong[:5],
            'trans_only_correct': trans_correct_native_wrong[:5],
            'native_only_correct': native_correct_trans_wrong[:5]
        }
        
        overlap['by_language'] = dict(by_language)
        
        return overlap
    
    def print_analysis(self, overlap: Dict):
        """Print detailed overlap analysis."""
        print("\n" + "="*80)
        print("ERROR OVERLAP ANALYSIS: DO PIPELINES FAIL ON SAME OR DIFFERENT QUESTIONS?")
        print("="*80)
        
        total = overlap['total_samples']
        
        print("\n📊 OVERALL BREAKDOWN:")
        print(f"   Total samples analyzed: {total}")
        print(f"\n   ✅ Both pipelines correct:     {overlap['both_correct']:4d} ({overlap['both_correct_pct']:5.1f}%)")
        print(f"   ❌ Both pipelines wrong:       {overlap['both_wrong']:4d} ({overlap['both_wrong_pct']:5.1f}%)")
        print(f"   🔄 Translation only correct:   {overlap['trans_only_correct']:4d} ({overlap['trans_only_correct_pct']:5.1f}%)")
        print(f"   🌍 Native only correct:        {overlap['native_only_correct']:4d} ({overlap['native_only_correct_pct']:5.1f}%)")
        
        print("\n" + "-"*80)
        print("🔍 ERROR OVERLAP ANALYSIS:")
        print(f"   Translation pipeline errors:   {overlap['translation_errors']}")
        print(f"   Native pipeline errors:        {overlap['native_errors']}")
        print(f"   Errors on SAME questions:      {overlap['both_wrong']} ({overlap['error_overlap_pct']:.1f}% of errors)")
        print(f"   Errors on DIFFERENT questions: {overlap['trans_only_correct'] + overlap['native_only_correct']}")
        
        print("\n" + "-"*80)
        print("🤝 COMPLEMENTARITY ANALYSIS:")
        
        complementary = overlap['trans_only_correct'] + overlap['native_only_correct']
        if complementary > 0:
            print(f"   Complementary correct answers: {complementary}")
            print(f"   → Translation helps on {overlap['trans_only_correct']} questions native fails")
            print(f"   → Native helps on {overlap['native_only_correct']} questions translation fails")
            
            balance = abs(overlap['trans_only_correct'] - overlap['native_only_correct'])
            if balance < complementary * 0.2:
                print(f"   ✅ Pipelines are BALANCED (similar complementary contributions)")
            elif overlap['trans_only_correct'] > overlap['native_only_correct']:
                print(f"   ⚖️  Translation pipeline MORE complementary (+{overlap['trans_only_correct'] - overlap['native_only_correct']} questions)")
            else:
                print(f"   ⚖️  Native pipeline MORE complementary (+{overlap['native_only_correct'] - overlap['trans_only_correct']} questions)")
        else:
            print(f"   ❌ No complementary correct answers (pipelines identical)")
        
        print("\n" + "-"*80)
        print("🎯 ENSEMBLE POTENTIAL:")
        print(f"   Current best accuracy:         {max(self.data['statistics']['translation']['accuracy'], self.data['statistics']['native']['accuracy']):.2f}%")
        print(f"   Perfect ensemble accuracy:     {overlap['ensemble_potential_accuracy']:.2f}%")
        print(f"   Potential gain:                +{overlap['ensemble_potential_gain']:.2f} percentage points")
        
        if overlap['ensemble_potential_gain'] > 5:
            print(f"\n   ✅ STRONG ensemble potential! Worth exploring voting/confidence methods")
        elif overlap['ensemble_potential_gain'] > 2:
            print(f"\n   ⚠️  MODERATE ensemble potential. May be worth exploring")
        else:
            print(f"\n   ❌ LOW ensemble potential. Pipelines too similar")
        
        print("\n" + "="*80)
        print("📍 PER-LANGUAGE BREAKDOWN:")
        print("="*80)
        print(f"\n{'Language':<10} {'Total':>6} {'Both✅':>7} {'Both❌':>7} {'Trans✅':>8} {'Native✅':>9} {'Comp.':>6}")
        print("-"*80)
        
        for lang, data in sorted(overlap['by_language'].items(), key=lambda x: -x[1]['total']):
            total_lang = data['total']
            both_correct = data['both_correct']
            both_wrong = data['both_wrong']
            trans_only = data['trans_only_correct']
            native_only = data['native_only_correct']
            complementary_pct = (trans_only + native_only) / total_lang * 100
            
            print(f"{lang:<10} {total_lang:6d} {both_correct:6d} {both_wrong:6d} {trans_only:7d} {native_only:8d} {complementary_pct:5.1f}%")
        
        print("\n" + "="*80)
        print("📝 SAMPLE QUESTIONS BY CATEGORY:")
        print("="*80)
        
        if overlap['examples']['trans_only_correct']:
            print("\n🔄 TRANSLATION CORRECT, NATIVE WRONG:")
            for i, ex in enumerate(overlap['examples']['trans_only_correct'][:3], 1):
                print(f"   {i}. [{ex['language']}] {ex['question']}")
                print(f"      Correct: {ex['correct_answer']} | Native predicted: {ex['native_predicted']}")
        
        if overlap['examples']['native_only_correct']:
            print("\n🌍 NATIVE CORRECT, TRANSLATION WRONG:")
            for i, ex in enumerate(overlap['examples']['native_only_correct'][:3], 1):
                print(f"   {i}. [{ex['language']}] {ex['question']}")
                print(f"      Correct: {ex['correct_answer']} | Translation predicted: {ex['trans_predicted']}")
        
        if overlap['examples']['both_wrong']:
            print("\n❌ BOTH PIPELINES WRONG (same errors):")
            for i, ex in enumerate(overlap['examples']['both_wrong'][:3], 1):
                print(f"   {i}. [{ex['language']}] {ex['question']}")
                print(f"      Correct: {ex['correct_answer']} | Trans: {ex['trans_predicted']} | Native: {ex['native_predicted']}")
        
        print("\n" + "="*80)
        print("💡 INTERPRETATION:")
        print("="*80)
        
        error_overlap_pct = overlap['error_overlap_pct']
        
        if error_overlap_pct > 80:
            print("\n   ❌ HIGH ERROR OVERLAP (>80%)")
            print("   → Pipelines fail on the SAME questions")
            print("   → Translation doesn't provide complementary signal")
            print("   → RECOMMENDATION: Focus on improving the model, not the pipeline")
        elif error_overlap_pct > 60:
            print("\n   ⚠️  MODERATE ERROR OVERLAP (60-80%)")
            print("   → Pipelines fail on SIMILAR questions")
            print("   → Some complementarity exists but limited")
            print("   → RECOMMENDATION: Ensemble MAY help slightly, but focus on model quality")
        else:
            print("\n   ✅ LOW ERROR OVERLAP (<60%)")
            print("   → Pipelines fail on DIFFERENT questions")
            print("   → Strong complementary signal!")
            print("   → RECOMMENDATION: Build an ensemble (voting/confidence-based)")
            print(f"   → Potential gain: +{overlap['ensemble_potential_gain']:.1f}% accuracy")
        
        print("\n" + "="*80)
    
    def save_analysis(self, overlap: Dict, output_file: str):
        """Save analysis to JSON."""
        output = {
            'analysis': 'Error Overlap Analysis',
            'source_file': self.data.get('experiment', 'unknown'),
            'overlap_metrics': overlap,
            'original_stats': self.data['statistics']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze error overlap between translation and native pipelines"
    )
    parser.add_argument(
        "--results_file",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_0_results.json",
        help="Path to experiment results JSON file"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_0_error_overlap_analysis.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    print(f"\n📂 Loading results from: {args.results_file}")
    analyzer = ErrorOverlapAnalyzer(args.results_file)

    
    print("\n🔍 Analyzing error overlap...")
    overlap = analyzer.analyze_overlap()
    
    analyzer.print_analysis(overlap)
    analyzer.save_analysis(overlap, args.output)
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()