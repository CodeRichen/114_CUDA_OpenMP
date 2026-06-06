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
            # 尋找「測資 ID」的開始標籤
            test_match = re.search(r"\[(測資 \d+)\]", line)
            if test_match:
                current_test = test_match.group(1)
                data[current_test] = ([], []) # (threads_list, times_list)
            
            # 擷取 Threads 數量與時間
            time_match = re.search(r"CPU Time \(Threads:\s+(\d+)\):\s+([\d.]+)\s+ms", line)
            if time_match and current_test:
                threads = int(time_match.group(1))
                time_ms = float(time_match.group(2))
                data[current_test][0].append(threads)
                data[current_test][1].append(time_ms)

    # 針對每一筆測資畫出一張獨立的圖
    for test_name, (threads, times) in data.items():
        if not threads:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # 繪製該測資的折線圖
        plt.plot(threads, times, marker='o', color='b', linewidth=2, label=test_name)

        # 標出每一個點的數值 (取小數點後兩位)
        for i, txt in enumerate(times):
            plt.annotate(f"{txt:.2f}", (threads[i], times[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

        plt.xlabel("Number of Threads")
        plt.ylabel("Execution Time (ms)")
        
        # 將 "測資 1" 轉換為 "Test Case 1" 避免中文字型顯示出問題
        en_title = test_name.replace("測資", "Test Case")
        plt.title(f"CPU Execution Time vs Number of Threads ({en_title})")
        
        plt.xticks(range(1, 13)) # X軸範圍 1 ~ 12
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # 設定檔案名稱 (例如：測資_1.png)
        safe_name = test_name.replace(" ", "_")
        out_img = f"cpu_time_threads_{safe_name}.png"
        
        plt.savefig(out_img, dpi=300)
        plt.close() # 關閉當前圖片，準備畫下一張
        print(f"圖表已成功生成並儲存為: {out_img}")

if __name__ == "__main__":
    main()
