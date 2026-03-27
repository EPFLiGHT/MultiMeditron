import os
import os

images_dir = "src/multimeditron/experts/evaluation_pipeline/xray_data/images"

broken_symlinks = []
total_symlinks = 0

for fname in os.listdir(images_dir):
    fpath = os.path.join(images_dir, fname)
    if os.path.islink(fpath):
        total_symlinks += 1
        target = os.readlink(fpath)
        target_path = target if os.path.isabs(target) else os.path.join(os.path.dirname(fpath), target)
        if not os.path.exists(target_path):
            broken_symlinks.append(fname)

print(f"Total symlinks: {total_symlinks}")
print(f"Broken symlinks: {len(broken_symlinks)}")
if total_symlinks > 0:
    valid_percent = 100 * (total_symlinks - len(broken_symlinks)) / total_symlinks
    print(f"Valid symlinks: {valid_percent:.2f}%")
else:
    print("No symlinks found.")


"""
if broken_symlinks:
    print("Broken symlinks (missing images):")
    for fname in broken_symlinks:
        print(fname)
else:
    print("All symlinks are valid.")
    """
