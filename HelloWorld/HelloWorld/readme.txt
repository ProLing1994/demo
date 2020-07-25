该工程展示了HelloWorld的调用方法。


一）运行方法：
1，配置工程，先选择ADKit选项，然后选择release模式
./rm_make.sh 
2，编译工程  
make all -j16
3，查看./release目录下编译结果


二）新项目可依照此项目结构进行改写，需要修改的文件如下：
./makefile 
该文件中MAKE_SUB_DIR变量要与同级目录的工程名（例如：app）对应。
./app/src.mk
该文件可根据项目需要改写：
#头文件路径，#库路径，#依赖静态库，#依赖动态库，#目标文件名，#发布库时待拷贝的文件


注意：demo工程仅实现了“可执行程序”输出到./release目录，新项目可通过修改 AIM_NAME 输出不同类型目标，AIM_NAME_EX 为扩展多输出使用。
输出为“库”时要据项目需要修改待输出的头文件路径（如：@cp ${PROJECT_DIR}/src/XXX.hpp ${RELEASE_DIR}/include）

为便于挂载调试时，可指定 BIN_OUTPUT_DIR 来拷贝所生成的目标文件（库或可执行程序）到指定目录。

板端挂载命令(xxx对应具体IP)
ifconfig eth0 192.168.51.xxx; route add default gw 192.168.51.251
mount -t nfs -o tcp,nolock 192.168.80.93:/home/workspace/RMAI/bin_output /mnt
cd /mnt/demo_test
设置依赖库“librmai_nnie.so”环境
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/demo_test
运行该目录下exe程序，具体参数请参考main函数，输入支持jpg/png等图片和yuv视频两种格式：
./demo_test  -i "./imgs/a.jpg" -c "./data/det3_inst.wk" -ct 0 -o "./b.jpg"
./demo_test  -i "./imgs/b.jpg" -c "./data/ssd_nnie_schoolbus_dp.wk" -ct 1 -o "./b.jpg"
./demo_test  -i "../videos/schoolbus/t1.yuv" -c "./data/ssd_nnie_schoolbus_dp.wk" -ct 1 -w 640 -h 360 -o "./result.yuv"