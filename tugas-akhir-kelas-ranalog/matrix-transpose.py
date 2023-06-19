import numpy as np
import multiprocessing
import datetime
import gc

import os
import sys
import pandas as pd

def process_index_element(chunk, n):
    # processor 2
    # chunk: [[nilai a[1], 1], [nilai a[5], 5]], n: 3
    new_chunk = []
    for e in chunk:
        # e: [nilai[1], 1]
        element = e[0]      # = nilai[1]
        index = e[1]        # = 1
        
        i = index // n      # = 1//3 = 0
        j = index % n       # = 1%3 = 1
        new_index = n*j + i # = 3*1 + 0 = 3
        
        new_e = [element, new_index]
        # new_e: [nilai[1], 3]
        new_chunk.append(new_e)
        
    return new_chunk

def transpose(matrix, cpu_num=1):
    time_start = datetime.datetime.now()
    n = matrix.shape[0]
    
    print("="*100)
    print("Process Start")
    print("CPU num          :", cpu_num)
    print("Matrix shape     :", matrix.shape)
    print("Matrix Input     :")
    print(matrix)
    
    flattened_matrix = np.array(matrix, dtype=np.int8).flatten()
    # del matrix
    
    chunks_parts = [([], n) for _ in range(cpu_num)]
    for index in range(n*n):
        element = flattened_matrix[index]
        chunk = [element, index]
        chunks_parts[index % cpu_num][0].append(chunk)
    
    # for chunk in chunks_parts:
    #     print(chunk)
    
    pool = multiprocessing.Pool(cpu_num)
    new_chunks_parts = pool.starmap(process_index_element, chunks_parts)
    
    # recreate matrix
    new_matrix = np.zeros(shape=(n,n), dtype=np.int8)
    for new_chunks_per_processor in new_chunks_parts:
        for e in new_chunks_per_processor:
            new_index = e[1]
            i = new_index // n
            j = new_index % n
            new_matrix[i, j] = e[0]
    
    # testing purpose. will throw an error and stop the program
    # execution if this fails
    np.testing.assert_array_equal(matrix.transpose(), new_matrix)
        
    time_finish = datetime.datetime.now()
    time_taken = time_finish - time_start
    
    np_time_start = datetime.datetime.now()
    matrix_t = matrix.transpose()
    np_time_taken = datetime.datetime.now() - np_time_start
    
    print()
    print("Process Finish")
    print("Time taken       :", time_taken)
    print("Time taken (np)  :", np_time_taken)
    print("Transpose matrix :")
    print(new_matrix)
    print()
    print("="*100)
    
    del matrix, new_matrix, chunks_parts, new_chunks_parts
    gc.collect() # garbage collector
    
    return time_taken

def pipeline_df(report):
    pd_report = pd.DataFrame(report)
    pd_report["matrix_shape"] = pd_report["matrix_shape"].apply(lambda x: x[0])
    pd_report["time_taken"] = pd_report["time_taken"].apply(lambda x: x.total_seconds())
    pd_report = pd_report.set_index("matrix_shape")
    cpu_num_1 = pd_report[pd_report["cpu_num"] == 1].drop(columns=["cpu_num"]).rename(columns={"time_taken": "1 processor"})
    cpu_num_2 = pd_report[pd_report["cpu_num"] == 2].drop(columns=["cpu_num"]).rename(columns={"time_taken": "2 processor"})
    cpu_num_3 = pd_report[pd_report["cpu_num"] == 3].drop(columns=["cpu_num"]).rename(columns={"time_taken": "3 processor"})
    cpu_num_4 = pd_report[pd_report["cpu_num"] == 4].drop(columns=["cpu_num"]).rename(columns={"time_taken": "4 processor"})
    pd_report = pd.concat([cpu_num_1, cpu_num_2, cpu_num_3, cpu_num_4], axis=1)
    return pd_report

if __name__ == "__main__":
    log_folder = os.path.join(os.getcwd(), "log")
    LOG_FILE = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_Log.log"
    os.makedirs(log_folder, exist_ok=True)
    LOG_FILE_PATH = os.path.join(log_folder, LOG_FILE)
    
    original_stdout = sys.stdout
    f = open(LOG_FILE_PATH, "a")
    sys.stdout = f
    
    # cpu_count = multiprocessing.cpu_count() # 4
    cpu_nums = [1, 2, 3, 4]
    # dims = [500, 1000, 1500, 2000, 2500, 3000]
    # dims = [x for x in range(10, 110, 10)] + [500] + [1000]
    dims = [500, 1000, 1500, 2000]
    
    report = []    
    for n in dims:
        for cpu_num in cpu_nums:
            matrix_input = np.random.randint(0, 100, size=[n, n], dtype=np.int8)
            time_taken = transpose(matrix_input, cpu_num)
            data = {
                "matrix_shape": matrix_input.shape,
                "cpu_num": cpu_num,
                "time_taken": time_taken
            }
            report.append(data)
    
    import pprint
    pprint.pprint(report)
    print()
    print(pipeline_df(report))
    
    sys.stdout = original_stdout