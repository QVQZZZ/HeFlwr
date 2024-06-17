param (
    [int]$num_rounds = 5,       # 客户端数量
    [int]$client_num = 8,       # 客户端数量
    [string]$dataset = "mnist", # 数据集
    [string]$partition = "iid", # 分区
    [float]$alpha = 0,          # alpha
    [int]$batch_size = 32       # batch_size
)

# 根据参数拼接生成输出文件名
$output_file = "fedrolex_rounds=${num_rounds}_dataset=${dataset}_clients=${client_num}_partition=${partition}_alpha=${alpha}_batch=${batch_size}.txt"

# 启动 server.py 并将输出重定向到生成的输出文件
Start-Process -FilePath "python" -ArgumentList "server.py --dataset $dataset --num_rounds $num_rounds" -RedirectStandardOutput $output_file -NoNewWindow

# 服务器地址
$server_address = "127.0.0.1:8080"

# 定义--p参数的值
$p_values = @("1/4", "1/2", "3/4", "1")

# 启动 client.py 的多个实例并放在后台运行
for ($cid = 1; $cid -le $client_num; $cid++) {
    # 计算当前客户端的--p参数值
    $p_value = $p_values[($cid - 1) % $p_values.Length]
    
    $arguments = "client.py --server_address $server_address --client_num $client_num --cid $cid --dataset $dataset --partition $partition --alpha $alpha --batch_size $batch_size --p $p_value"
    Start-Process -FilePath "python" -ArgumentList $arguments -NoNewWindow
}
