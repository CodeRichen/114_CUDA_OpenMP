import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ─────────────────────────────────────────────
# 解析輸出檔
# ─────────────────────────────────────────────

def parse_output(filepath: str):
    """
    回傳 list of dict：
    [
      { 'id': 1, 'label': 'T:(3750x4320) S:(3x3)', 'times': [274.44, 120.63, ...] },
      ...
    ]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 以 "================" 切分每筆測資
    blocks = re.split(r'={40,}', text)

    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 取測資標頭，例如 [測資 1] T:(3750x4320) S:(3x3)
        header_m = re.search(r'\[測資\s*(\d+)\]\s*(T:\(\d+x\d+\)\s*S:\(\d+x\d+\))', block)
        if not header_m:
            continue
        tc_id    = int(header_m.group(1))
        tc_label = header_m.group(2).strip()

        # 取各執行緒時間
        time_matches = re.findall(
            r'CPU Time \(Threads:\s*(\d+)\):\s*([\d.]+)\s*ms', block
        )
        if not time_matches:
            continue

        times = [None] * 12
        for t_str, ms_str in time_matches:
            idx = int(t_str) - 1          # 0-based
            if 0 <= idx < 12:
                times[idx] = float(ms_str)

        results.append({
            'id':    tc_id,
            'label': tc_label,
            'times': times,
        })

    results.sort(key=lambda x: x['id'])
    return results

# ─────────────────────────────────────────────
# 畫圖
# ─────────────────────────────────────────────

COLORS   = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
THREADS  = list(range(1, 13))

def plot_all(results, out_dir: str = '.'):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 各自獨立的折線圖 ──────────────────────
    for i, r in enumerate(results):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        color = COLORS[i % len(COLORS)]

        times = r['times']
        valid_x = [t for t, v in zip(THREADS, times) if v is not None]
        valid_y = [v for v in times if v is not None]

        ax.plot(valid_x, valid_y,
                marker='o', linewidth=2, markersize=6,
                color=color, label=r['label'])

        # 標最小值
        min_y = min(valid_y)
        min_x = valid_x[valid_y.index(min_y)]
        ax.annotate(f'min {min_y:.2f} ms',
                    xy=(min_x, min_y),
                    xytext=(min_x + 0.4, min_y + (max(valid_y) - min_y) * 0.08),
                    fontsize=8, color='black',
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1))

        ax.set_title(f'Test Case {r["id"]}  |  {r["label"]}', fontsize=12, pad=10)
        ax.set_xlabel('Number of Threads', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_xticks(THREADS)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = os.path.join(out_dir, f'tc{r["id"]}_thread_time.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'  已儲存：{fname}')

    # ── 2. 所有測資合併於同一張圖 ────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, r in enumerate(results):
        color = COLORS[i % len(COLORS)]
        times = r['times']
        valid_x = [t for t, v in zip(THREADS, times) if v is not None]
        valid_y = [v for v in times if v is not None]
        ax.plot(valid_x, valid_y,
                marker='o', linewidth=2, markersize=5,
                color=color, label=f'TC{r["id"]} {r["label"]}')

    ax.set_title('All Test Cases – Thread Count vs Time', fontsize=13, pad=10)
    ax.set_xlabel('Number of Threads', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_xticks(THREADS)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    fname_all = os.path.join(out_dir, 'all_tc_thread_time.png')
    fig.savefig(fname_all, dpi=150)
    plt.close(fig)
    print(f'  已儲存（合併圖）：{fname_all}')

# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    INPUT_FILE = 'output_cpu_only.txt'   # ← 修改成你的輸出檔路徑
    OUTPUT_DIR = 'plots'

    print(f'讀取：{INPUT_FILE}')
    results = parse_output(INPUT_FILE)
    print(f'解析到 {len(results)} 筆測資')
    plot_all(results, out_dir=OUTPUT_DIR)
    print('完成。')