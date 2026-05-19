import re
import matplotlib.pyplot as plt
import numpy as np

def parse_output(filename):
    data = {}
    current_test = None
    current_block = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Match Test Case
            # [測資 1] T:(3750x4320) S:(3x3)
            test_match = re.match(r'^\[測資\s*(\d+)\]\s*(.*)$', line)
            if test_match:
                test_id = f"Task_{test_match.group(1)}"
                current_test = test_id
                data[current_test] = {'info': test_match.group(2), 'blocks': []}
                continue
            
            # Match Block Size
            block_match = re.match(r'^▶ Block Size:\s*(\d+)x\s*(\d+)$', line)
            if block_match:
                current_block = f"{block_match.group(1)}x{block_match.group(2)}"
                continue
                
            # Match Average Time
            time_match = re.match(r'^平均時間 - Global:\s*([0-9.]+)\s*ms\s*\|\s*Opt\(T_share, S_const\):\s*([0-9.]+)\s*ms$', line)
            if time_match and current_test and current_block:
                global_time = float(time_match.group(1))
                opt_time = float(time_match.group(2))
                data[current_test]['blocks'].append({
                    'block': current_block,
                    'global': global_time,
                    'opt': opt_time
                })
                current_block = None
                
    return data

def main():
    data = parse_output('output.txt')
    
    for test_id, test_data in data.items():
        # print(f"\n================ {test_id} {test_data['info']} ================")
        blocks = test_data['blocks']
        
        blocks_sorted = sorted(blocks, key=lambda x: x['opt'])
        
        print("【依 Globle 執行時間排名】:")
        for idx, b in enumerate(blocks_sorted, 1):
            print(f"  {idx:2d}. {b['block']:>9} -> Opt: {b['opt']:8.4f} ms | Global: {b['global']:8.4f} ms")
            
        # Draw chart for each test case
        # labels = [b['block'] for b in blocks]
        # opt_times = [b['opt'] for b in blocks]
        # global_times = [b['global'] for b in blocks]
        labels = [b['block'] for b in blocks_sorted]
        opt_times = [b['opt'] for b in blocks_sorted]
        global_times = [b['global'] for b in blocks_sorted]
        
        x = np.arange(len(labels))
        width = 0.5
        
        fig, ax = plt.subplots(figsize=(6, 6))
        # rects1 = ax.bar(x - width/2, global_times, width, label='Global Memory', color='#1f77b4')
        rects2 = ax.bar(x, opt_times, width, label='Time', color='#3399ff')
        
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title(f'{test_id} Performance Analysis\n{test_data["info"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=360)
        ax.legend()
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        # autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        filename = f'./graphs/{test_id}_analysis.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"-> 直方圖已儲存至 {filename}")

    # ========================================================
    # 新增：畫出所有測資 (4 個 Test Case) 的平均比較圖
    # ========================================================
    avg_data_map = {}
    for test_id, test_data in data.items():
        for b in test_data['blocks']:
            blk = b['block']
            if blk not in avg_data_map:
                avg_data_map[blk] = {'opt': [], 'global': []}
            avg_data_map[blk]['opt'].append(b['opt'])
            avg_data_map[blk]['global'].append(b['global'])
            
    # 計算各 Block Size 的平均
    avg_list = []
    for blk, vals in avg_data_map.items():
        avg_list.append({
            'block': blk,
            'opt_avg': np.mean(vals['opt']),
            'global_avg': np.mean(vals['global'])
        })
        
    # 根據 Opt 平均時間排序
    avg_sorted = sorted(avg_list, key=lambda x: x['opt_avg'])
    
    avg_labels = [b['block'] for b in avg_sorted]
    avg_opt_times = [b['opt_avg'] for b in avg_sorted]
    avg_global_times = [b['global_avg'] for b in avg_sorted]
    
    x_avg = np.arange(len(avg_labels))
    width_avg = 0.35
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 畫 Global (平均) 與 Opt/Share (平均)
    rects_global = ax2.bar(x_avg - width_avg/2, avg_global_times, width_avg, label='Global Memory (Avg of 4 Tasks)', color='#ff8000')
    rects_opt = ax2.bar(x_avg + width_avg/2, avg_opt_times, width_avg, label='Shared Memory (Avg of 4 Tasks)', color='#3399ff')
    
    ax2.set_ylabel('Average Execution Time (ms)')
    ax2.set_title('All Tasks Average Performance Analysis')
    ax2.set_xticks(x_avg)
    ax2.set_xticklabels(avg_labels, rotation=360)
    ax2.legend()
    
    def autolabel_avg(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel_avg(rects_global)
    autolabel_avg(rects_opt)
    
    fig2.tight_layout()
    import os
    if not os.path.exists('./graphs'):
        os.makedirs('./graphs')
    avg_filename = './graphs/All_Tasks_Average_analysis.png'
    plt.savefig(avg_filename, dpi=300)
    plt.close()
    print(f"-> 綜合平均直方圖已儲存至 {avg_filename}")

if __name__ == '__main__':
    main()

