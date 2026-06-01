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

    # 繪製圖表
    plt.figure(figsize=(10, 6))
    for test, (threads, times) in data.items():
        if threads: # 確定有抓到數據
            plt.plot(threads, times, marker='o', label=test)

    # 設定圖表標籤與格式
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (ms) - Log Scale")
    plt.title("CPU Execution Time vs Number of Threads (OpenMP)")
    plt.xticks(range(1, 13)) # X 軸為 1 到 12
    # 由於各測資(如測資 4 與 3)的時間落差可能高達上萬倍，建議使用對數 (Log) 坐標軸
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # 儲存與顯示結果
    out_img = "cpu_time_threads.png"
    plt.savefig(out_img, dpi=300)
    print(f"圖表已成功生成並儲存為: {out_img}")

if __name__ == "__main__":
    main()
