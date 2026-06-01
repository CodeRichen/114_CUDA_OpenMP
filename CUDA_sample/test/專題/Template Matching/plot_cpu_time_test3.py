import re
import matplotlib.pyplot as plt
import os

def main():
    log_file = "output.txt"
    if not os.path.exists(log_file):
        print(f"找不到檔案: {log_file}")
        return

    data = {}
    current_test = ""
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            # 尋找「測試資料」的開始標籤
            test_match = re.search(r"\[(測資 \d+)\]", line)
            if test_match:
                current_test = test_match.group(1)
                data[current_test] = ([], []) # (threads_list, times_list)
            
            # 使用 Regex 解析 CPU Time 與 Threads 數量
            time_match = re.search(r"CPU Time \(Threads:\s+(\d+)\):\s+([\d.]+)\s+ms", line)
            if time_match and current_test:
                threads = int(time_match.group(1))
                time_ms = float(time_match.group(2))
                data[current_test][0].append(threads)
                data[current_test][1].append(time_ms)

    # 檢查是否有抓到「測資 3」
    target_test = "測資 3"
    if target_test not in data or not data[target_test][0]:
        print(f"找不到 {target_test} 的數據")
        return

    threads, times = data[target_test]

    # 繪製圖表
    plt.figure(figsize=(8, 5))
    
    # 這裡將標籤改為英文，避免 matplotlib 顯示中文產生亂碼警告
    plt.plot(threads, times, marker='o', color='b', label='Test Case 3', linewidth=2)

    # 標上每個點的數據
    for i, txt in enumerate(times):
        plt.annotate(f"{txt:.1f}", (threads[i], times[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 設定圖表標籤與格式
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (ms)")
    plt.title("CPU Execution Time vs Number of Threads (Test Case 3)")
    plt.xticks(range(1, 13)) # X 軸為 1 到 12
    # 由於只有單一數據，使用線性軸能更好看出加速比例
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 儲存與顯示結果
    out_img = "cpu_time_threads_test3.png"
    plt.savefig(out_img, dpi=300)
    print(f"圖表已成功生成並儲存為: {out_img}")

if __name__ == "__main__":
    main()
