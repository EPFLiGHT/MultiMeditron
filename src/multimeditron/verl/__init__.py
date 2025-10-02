import ray
import random
from omegaconf import OmegaConf

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self,
            cfg,
            trust_remote_code: bool = False,
            verbose: bool = False,
            dryrun: bool = False):
        from transformers import AutoTokenizer

        if verbose:
            from pprint import pprint
            pprint("Final Merged Configuration:")
            pprint(cfg.model_dump())

        # Instantiate tokenizer
        print("Instantiating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.actor_rollout_ref.model.path,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        
        # For each strategy, we would have different training loops
        # assert cfg.actor_rollout_ref.model.strategy == cfg.critic.model.strategy, "Currently only support same strategy for actor and critic"
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

        # Setup Ray
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # We should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data

        # If we want reward model
        # if config.reward_model.enable:
        #     if config.reward_model.strategy == "fsdp":
        #         from verl.workers.fsdp_workers import RewardModelWorker
        #     elif config.reward_model.strategy == "megatron":
        #         from verl.workers.megatron_workers import RewardModelWorker
        #     else:
        #         raise NotImplementedError
        #     role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        #     mapping[Role.RewardModel] = global_pool_id
        
        # Reference model
        # if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        if cfg.algorithm.use_kl_in_reward:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = cfg.reward_model.get("reward_manager", "naive")
        match reward_manager_name:
            case 'batch':
                from verl.workers.reward_manager import BatchRewardManager
                reward_manager_cls = BatchRewardManager

            case 'naive':
                from verl.workers.reward_manager import NaiveRewardManager
                reward_manager_cls = NaiveRewardManager
            
            case 'prime':
                from verl.workers.reward_manager import PrimeRewardManager
                reward_manager_cls = PrimeRewardManager

            case 'dapo':
                from verl.workers.reward_manager import DAPORewardManager
                reward_manager_cls = DAPORewardManager

            case 'async_dapo':
                raise NotImplementedError
            
            case _:
                raise NotImplementedError(f"Reward manager {reward_manager_name} not implemented")

        # compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=cfg.data.reward_fn_key,
            max_resp_len=cfg.actor_rollout_ref.rollout.response_length,
            overlong_buffer_cfg=cfg.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=cfg.data.reward_fn_key,
            max_resp_len=cfg.actor_rollout_ref.rollout.response_length,
            overlong_buffer_cfg=cfg.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        
        cfg = OmegaConf.create(cfg)
        OmegaConf.resolve(cfg)
        # trainer_config = OmegaConf.to_container(trainer_config, resolve=True)
        # OmegaConf.resolve(trainer_config)
        # print(trainer_config)
        # return

        trainer = RayPPOTrainer(
            config=cfg,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            collate_fn=None,
        )
        trainer.init_workers()

        if not dryrun:
            trainer.fit()
        else:
            print("Dry run complete. Exiting without training.")
    
def collate_fn(x): 
    return x

def compute_score(data_source, solution_str, ground_truth, extra_info):
    # print(extra_info)
    # if extra_info is None or "question" not in extra_info or "url" not in extra_info:
    #     raise ValueError("Extra info is required and must contain 'question' and 'url'")
    
    # do_print = False
    # if random.randint(0, 512) == 1:  
    #     do_print = True
    # if do_print:
    #     print(f"Response Case: {solution_str}, Question: {extra_info['question']}, GT: {ground_truth}")

    response = solution_str
    response_lower = response.lower()
    score = response_lower.count("a") / len(response_lower) if len(response_lower) > 0 else 0
    print(f"Score: {score}")

    return {
        "score": score,
        "acc": 0.0,
        "pred": "Maybe",
    }