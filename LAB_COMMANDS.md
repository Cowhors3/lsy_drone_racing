# Lab commands

```bash
python scripts/deploy.py --config level2.toml
```

```bash
E2E_POLICY=policy_backout.npz python scripts/deploy.py --config level2.toml
```

```bash
E2E_POLICY=policy_fast.npz python scripts/deploy.py --config level2.toml
```

```bash
ls -la logs/
```

```bash
git config user.name "Cowhors3"
git config user.email "Cowhors3@users.noreply.github.com"
git add -f logs
git commit -m "Add lab flight logs"
git push
```
