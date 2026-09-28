# Second-paper Terminal Commands

## 进入项目

```bash
cd /home/gxy/code/Second-paper
```

## 激活环境

```bash
conda activate gume
```

## 查看 GPU

```bash
nvidia-smi
```

## Baby：MSCA

```bash
python src/main.py -m MSCA -d baby --stage msca --gpu-id 0
```

## Baby：MSCA + CoLiftRec

```bash
python src/main.py -m MSCA -d baby --stage coliftrec --gpu-id 0
```

## Baby：MSCA + CoLiftRec + Diffusion

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0
```

## Sports：MSCA

```bash
python src/main.py -m MSCA -d sports --stage msca --gpu-id 0
```

## Sports：MSCA + CoLiftRec

```bash
python src/main.py -m MSCA -d sports --stage coliftrec --gpu-id 0
```

## Sports：MSCA + CoLiftRec + Diffusion

```bash
python src/main.py -m MSCA -d sports --stage full --gpu-id 0
```

## Electronics：MSCA

```bash
python src/main.py -m MSCA -d elec --stage msca --gpu-id 0
```

## Electronics：MSCA + CoLiftRec

```bash
python src/main.py -m MSCA -d elec --stage coliftrec --gpu-id 0
```

## Electronics：MSCA + CoLiftRec + Diffusion

```bash
python src/main.py -m MSCA -d elec --stage full --gpu-id 0
```

## Baby Full Smoke

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0 --smoke
```

## Sports Full Smoke

```bash
python src/main.py -m MSCA -d sports --stage full --gpu-id 0 --smoke
```

## Electronics Full Smoke

```bash
python src/main.py -m MSCA -d elec --stage full --gpu-id 0 --smoke
```

## Baby Full Dry-run

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0 --dry-run
```

## Sports Full Dry-run

```bash
python src/main.py -m MSCA -d sports --stage full --gpu-id 0 --dry-run
```

## Electronics Full Dry-run

```bash
python src/main.py -m MSCA -d elec --stage full --gpu-id 0 --dry-run
```

## Baby Full 后台运行

```bash
nohup python -u src/main.py -m MSCA -d baby --stage full --gpu-id 0 > log/nohup-baby-full.out 2>&1 &
```

## Sports Full 后台运行

```bash
nohup python -u src/main.py -m MSCA -d sports --stage full --gpu-id 0 > log/nohup-sports-full.out 2>&1 &
```

## Electronics Full 后台运行

```bash
nohup python -u src/main.py -m MSCA -d elec --stage full --gpu-id 0 > log/nohup-elec-full.out 2>&1 &
```

## 查看最近生成的主日志

```bash
ls -lt log | head
```

## 实时查看最新主日志

```bash
tail -f "$(ls -t log/*.log | head -1)"
```

## 实时查看 Baby nohup 输出

```bash
tail -f log/nohup-baby-full.out
```

## 实时查看 Sports nohup 输出

```bash
tail -f log/nohup-sports-full.out
```

## 实时查看 Electronics nohup 输出

```bash
tail -f log/nohup-elec-full.out
```

## 查看当前 Python 实验进程

```bash
ps -ef | grep "src/main.py" | grep -v grep
```

## 查看 GPU 上的进程

```bash
nvidia-smi
```

## 查看 Git 状态

```bash
git status
```

## 查看最近提交

```bash
git log --oneline -10
```

## 拉取 GitHub 最新代码

```bash
git pull origin main
```

## 推送到 GitHub

```bash
git push origin main
```

## 查看当前 HEAD

```bash
git rev-parse HEAD
```

## 查看远端 main

```bash
git ls-remote origin refs/heads/main
```
