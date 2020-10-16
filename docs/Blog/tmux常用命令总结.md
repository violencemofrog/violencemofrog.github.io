# tmux的组成

每次打开一个tmux，都会默认打开一个session，该session下又会默认打开一个window，该window下又会默认打开一个pane

# session（会话）相关命令

* `tmux new -s session_name`：新建会话
* `tmux detach/Ctrl+b d`：在tmux中输入该命令，使当前会话转入后台
* `tmux ls/Ctrl+b s`：查看后台所有会话
* `tmux attach -t n/session_name`：通过编号或会话名称重新接入会话
* `tmux kill-session -t n/session_name`：终止会话
* `tmux switch -t n/session_name`：切换会话
* `tmux rename-session -t n/session_name`/Ctrl+b $：重命名会话

# window（窗口）相关命令

* `tmux new-window -n window_name/Ctrl+b c`：创建新窗口
* `tmux select-window -t n/window_name`：切换窗口
* `Ctrl+b p/n/b+n`：切换到上/下/第n个窗口
* `tmux rename-window new_window_name/Ctrl+b ,`为当前窗格重命名
* `Ctrl+b w`：从列表中切换窗格

# pane（窗格）相关命令

* `tmux split-window (-h)`：分裂为上下（左右）两个窗格
* `Ctrl+b %/"`：：分裂为上下（左右）两个窗格
* `Ctrl+b 方向键`：焦点移动到其他窗格
* `Ctrl+b {/}`：当前窗格和上一个/下一个窗格交换位置
* `Ctrl+b x`：关闭当前窗格
* `Ctrl+b !`：将当前窗格拆分为独立窗口
* `Ctrl+b Ctrl+方向键`：按指定方向调整窗格大小
* `Ctrl+b q`：显示窗格编号
