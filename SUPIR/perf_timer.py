import io
import time
import psutil
import matplotlib.pyplot as plt
import torch
from PIL import Image

vram_supported = False
if torch.cuda.is_available():
    vram_supported = True


class PerfTimer:
    def __init__(self, print_log=False):
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.base_category = ''
        self.print_log = print_log
        self.subcategory_level = 0
        self.ram_records = []
        self.vram_records = []
        self.time_points = []

    def get_ram_usage(self):
        return psutil.Process().memory_info().rss / (1024 ** 3)  # GB

    def get_vram_usage(self):
        if vram_supported:  # Ensure vram_supported is defined and correctly determines if VRAM usage can be checked
            torch.cuda.synchronize()  # Wait for all kernels in all streams on a CUDA device to complete
            info = torch.cuda.memory_stats()  # Get detailed CUDA memory stats
            used = info['allocated_bytes.all.peak']  # Get peak allocated bytes
            return used / (1024 ** 3)  # Convert bytes to GB
        return 0

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def add_time_to_record(self, category, amount):
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    def record(self, category, extra_time=0, disable_log=False):
        e = self.elapsed()
        ram_usage = self.get_ram_usage()
        vram_usage = self.get_vram_usage()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time
        self.time_points.append(self.total)
        self.ram_records.append(ram_usage)
        self.vram_records.append(vram_usage)

        if self.print_log and not disable_log:
            print(
                f"{'  ' * self.subcategory_level}{category}: done in {e + extra_time:.3f}s, RAM: {ram_usage:.2f}GB, VRAM: {vram_usage:.2f}GB")

    def make_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ram_records, label='RAM Usage (GB)', marker='o')
        if vram_supported:
            plt.plot(self.time_points, self.vram_records, label='VRAM Usage (GB)', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Usage (GB)')
        plt.title('Performance Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        image = Image.open(img_buf)
        img_buf.close()

        return image

    def summary(self):
        res = f"{self.total:.1f}s"

        additions = [(category, time_taken) for category, time_taken in self.records.items() if
                     time_taken >= 0.1 and '/' not in category]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def dump(self):
        return {'total': self.total, 'records': self.records, 'ram_usage': self.ram_records,
                'vram_usage': self.vram_records, 'time_points': self.time_points}

    def reset(self):
        self.__init__()
