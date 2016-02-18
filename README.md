centos6.2升级内核

一、查看当前内核版本 
uname -a #查看当前内核版本 
2.6.32-220.13.1.el6.x86_64 #1 SMP Tue Apr 17 23:56:34 BST 2012 x86_64 x86_64 x86_64 GNU/Linux 
 
二、下载待安装的kernel并解压 
wget http://www.kernel.org/pub/linux/kernel/v3.0/linux-3.6.8.tar.bz2 
tar -jxf linux-3.6.8.tar.bz2 
mv linux-3.6.8.tar.bz2 /usr/src/kernels/ 
cd /usr/src/kernels/linux-3.6.8.tar.bz2/ 
 
 
三、安装编译内核所需的工具 
yum install ncurses-devel 
yum -y install gcc automake autoconf libtool make 
 
四、内核编译安装过程 
#make mrproper #首次编译可以省略该步，如果之前在此目录编译过，该命令可以删除之前编译所生成的文件和配置文件，备份文件 
cp /boot/config-2.6.32-220.13.1.el6.x86_64 /usr/src/kernels/linux-3.6.8/.config #在当前内核参数的基础上来，选择我们想要增删的参数进行编译，这点很重要，否则可能有各种奇怪的问题 
make menuconfig #在菜单模式下选择需要编译的内核模块，可以参考内核编译配置选项：http://lamp.linux.gov.cn/Linux/kernel_options.html   
 
 
make bzImage #生成内核文件(漫长等待，我用了20分钟) 
 
 
make modules #编译模块(非常慢，耐心等待) 
 
 
make modules_install #安装模块 
 
 
make install #安装 
 
 
 
 
五、设置从新内核启动 
vi /boot/grub/grub.conf 
将 default=1 改为 default=0 
 
 
 
六、重启系统并查看内核版本 
shutdown –f -r now 
正常登录后，查看内核版本 
uname –r 
 
 
OK，内核升级成功。
