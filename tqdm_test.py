import time

from tqdm import tqdm

bar = tqdm(range(10000), dynamic_ncols=True)
for x in bar:
    bar.set_description("somethinglong_" * 15, desc_short="ronan")
    time.sleep(0.001)
