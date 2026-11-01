import argparse, yaml, wandb

def main():
    p=argparse.ArgumentParser(); p.add_argument("--create",action="store_true")
    p.add_argument("--sweep-id"); p.add_argument("--count",type=int,default=30)
    p.add_argument("--config",default="configs/sweep.yaml"); a=p.parse_args()
    if a.sweep_id: sid=a.sweep_id
    elif a.create:
        cfg=yaml.safe_load(open(a.config,encoding="utf-8"))
        sid=wandb.sweep(cfg,project="cs6886w-vgg6")
    else: raise SystemExit("Use --create or --sweep-id.")
    print("Sweep ID:",sid); wandb.agent(sid,count=a.count,project="cs6886w-vgg6")
if __name__=="__main__": main()
