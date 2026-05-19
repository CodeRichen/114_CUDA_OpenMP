import re

with open("/home/A1125515/114/CUDA_sample/test/專題/Template Matching/test_all.cu", "r") as f:
    text = f.read()

# Replace TestCase struct
old_struct = """typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
} TestCase;"""

new_struct = """typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
    int exp_r1, exp_c1;
    int exp_r2, exp_c2;
} TestCase;"""

text = text.replace(old_struct, new_struct)

# Replace tests array
old_tests = """    TestCase tests[] = {
        {1, "test data/1/T1_3750_4320.txt", 3750, 4320, "test data/1/S1_3_3.txt", 3, 3},
        {2, "test data/2/T2_7750_1320.txt", 7750, 1320, "test data/2/S2_5_5.txt", 5, 5},
        {3, "test data/3/T3_8140_9925.txt", 8140, 9925, "test data/3/S3_3_3.txt", 3, 3},
        {4, "test data/4/T4_50_50.txt", 50, 50, "test data/4/S4_5_5.txt", 5, 5}
    };"""

new_tests = """    TestCase tests[] = {
        {1, "test data/1/T1_3750_4320.txt", 3750, 4320, "test data/1/S1_3_3.txt", 3, 3, 581, 1280, -1, -1},
        {2, "test data/2/T2_7750_1320.txt", 7750, 1320, "test data/2/S2_5_5.txt", 5, 5, 7691, 688, -1, -1},
        {3, "test data/3/T3_8140_9925.txt", 8140, 9925, "test data/3/S3_3_3.txt", 3, 3, 2800, 6, 4653, 4239},
        {4, "test data/4/T4_50_50.txt", 50, 50, "test data/4/S4_5_5.txt", 5, 5, 18, 17, -1, -1}
    };"""

text = text.replace(old_tests, new_tests)

# Insert checking logic
insert_point = text.find('        printf("平均時間 - Global: %8.4f ms | Opt(T_share, S_const): %8.4f ms\\n", \n               total_ms_global / 3.0f, total_ms_optimized / 3.0f);')
if insert_point != -1:
    end_insert_point = text.find('}\n', insert_point) + 1
    
    validation_code = """
        // ----------------- 驗證環節 -----------------
        CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        float max_pcc = -2.0f;
        for (int i = 0; i < out_size; i++) {
            if (!isnan(h_pcc_out[i]) && h_pcc_out[i] > max_pcc) {
                max_pcc = h_pcc_out[i];
            }
        }
        
        bool found1 = false;
        bool found2 = false;
        for (int i = 0; i < out_size; i++) {
            if (fabs(h_pcc_out[i] - max_pcc) < 1e-4) {
                int r = i / out_c;
                int c = i % out_c;
                if (r == tc.exp_r1 && c == tc.exp_c1) found1 = true;
                if (tc.exp_r2 != -1 && r == tc.exp_r2 && c == tc.exp_c2) found2 = true;
            }
        }
        
        bool is_valid = found1;
        if (tc.exp_r2 != -1) is_valid = found1 && found2;

        if (is_valid) {
            printf("  ✔️ 驗證結果: 合法\n");
        } else {
            printf("  ❌ 驗證結果: 無效 (PCC最高點位置不符合預期)\n");
        }
        // ------------------------------------------
"""
    # put the new code right before the closing brace of the loop
    text = text[:end_insert_point-1] + validation_code + text[end_insert_point-1:]

# Also remove the out of loop memcpy since we do it inside loop now
text = text.replace("""    // 在所有測資 Block Size 組合跑完後，只 Memcpy 最後一次的結果出來做驗證即可
    CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ssd_out, d_ssd, out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
""", """    // 最後一次測試的結果已經複製到 h_pcc_out 中
""")

with open("/home/A1125515/114/CUDA_sample/test/專題/Template Matching/test_all.cu", "w") as f:
    f.write(text)

