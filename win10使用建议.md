#### 一. 调整桌面图标间距

**具体方法如下:**

1、按Win+R，然后输入regedit，回车进入注册表编辑器。

2、找到这里：HKEY_CURRENT_USER\Control Panel\Desktop\WindowMetrics

3、桌面图标水平间距：

​	IconSpacing，默认值大概是-1125，想缩小间距就改大一些，比如-800。

​	桌面图标垂直间距：

​	IconVerticalSpacing，同上。

4、修改后可能需要注销或重启才能生效。