#### DOTA坐标转换

##### 1. 修改tools/test.py文件，保存测试数据名， ==修改文件路径==即可

![](https://github.com/Alexanderisgod/PicBed/blob/main/20220602215746.png?raw==true)



##### 2. 输出测试结果， --out 保存文件地址

> #### 单卡测试
> python 	tools/test.py 	/root/mmrotate/configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py 	/root/mmrotate/work_dirs/r3det_kfiou_ln_r50_fpn_1x_dota_oc/epoch_12.pth 	==--out out.pkl==

​		可使用 --show-dir  your_dir_to_save_results	保存结果。如图;

​		![](https://github.com/Alexanderisgod/PicBed/blob/main/20220602220417.png?raw==true)



##### 3. 运行脚本对输出转换——4个顶点, 修改以上三个==文件路径即可==

![](https://github.com/Alexanderisgod/PicBed/blob/main/20220602220147.png?raw==true)

##### 4. 使用Dota devkit进行评测    mAP

==dota devkit安装==

> git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
>
>  sudo apt-get install swig
>
> cd Dota_devkit
>
>  swig -c++ -python polyiou.i
>
> python setup.py build_ext --inplace

只修改 dota_evaluation_task1.py 以下两点即可

![](https://github.com/Alexanderisgod/PicBed/blob/main/20220602220623.png?raw==true)



##### 5.  自行可视化结果

drawrect.py



