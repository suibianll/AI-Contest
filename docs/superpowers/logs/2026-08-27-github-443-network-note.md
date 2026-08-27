# GitHub 443 间歇重置诊断与解法（工程记录）

日期：2026-08-27
环境：Windows + PowerShell 7，仓库 `AI-Contest`（origin:
`https://github.com/suibianll/AI-Contest.git`），git 全局配置无代理。

## 现象

`git push` 间歇性失败，两种报错交替出现：

```text
fatal: unable to access 'https://github.com/suibianll/AI-Contest.git/':
Recv failure: Connection was reset

fatal: unable to access '...': Failed to connect to github.com port 443
after 21115 ms: Could not connect to server
```

同一会话内 e4f4c7b 推送成功、7c2227a 推送失败；连重试 3 次均失败。
**结论：重试解决不了，需要换传输路径。**

## 诊断（PowerShell 实测）

| 检测项 | 结果 | 含义 |
|---|---|---|
| `Test-NetConnection github.com -Port 443` | TCP 可建立（20.205.243.166） | 443 端口本身未被墙 |
| `git push`（同一时刻） | TLS 握手被重置（Recv failure） | **SNI 阻断特征**：TCP 通但 HTTPS 握手被干扰 |
| `Test-NetConnection ssh.github.com -Port 443` | TCP 可达（20.205.243.160） | SSH-over-443 通道可用 |
| `Test-NetConnection 127.0.0.1 -Port 7890` | 无监听 | 本机无 Clash 类代理在跑 |

判定：属于对 `github.com` HTTPS 的间歇性 SNI 干扰（GFW 典型行为），
非本地防火墙/代理配置问题。TCP 三次握手成功但 TLS ClientHello 后被
RST，间歇触发。

## 解法（按可靠性排序）

### 方案 1（推荐）：SSH 走 443 端口，一次配置永久生效

SSH 协议不做 TLS 握手，不受 SNI 重置影响：

```powershell
# 1. 生成密钥（已有则跳过）
ssh-keygen -t ed25519 -C "2990350431@qq.com"

# 2. 把公钥贴到 GitHub → Settings → SSH and GPG keys（需手动一次）
Get-Content ~\.ssh\id_ed25519.pub

# 3. 写 ~\.ssh\config
#    Host github.com
#        HostName ssh.github.com
#        Port 443
#        User git

# 4. 换 remote（本项目操作）
git remote set-url origin git@github.com:suibianll/AI-Contest.git

# 5. 验证
ssh -T git@github.com   # 出现 "Hi suibianll!" 即成功
```

注意：换 SSH 后不再走 credential manager；`ssh-keygen` 若覆盖旧密钥
会导致旧部署失效，优先复用已有密钥。

### 方案 2：本地代理（有 Clash/v2ray 时）

```powershell
git config --global http.proxy http://127.0.0.1:7890
# 取消：git config --global --unset http.proxy
```

### 方案 3：Gitee 镜像兜底（国内保底同步）

本机已有 gitee.com 凭据配置，可直接加镜像 remote：

```powershell
git remote add gitee https://gitee.com/<用户名>/AI-Contest.git
git push gitee master
```

## 相关网络经验（本项目已验证）

- Hugging Face 模型下载用镜像：`HF_ENDPOINT=https://hf-mirror.com`；
- GitHub HTTPS 在本网络环境**只能视为尽力而为**：推送失败先 `git status`
  确认本地提交已落库（无丢失风险），再择机重推或换通道。
