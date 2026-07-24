from pathlib import Path
import torch


model_dir = Path(r"C:\Users\22825\Desktop\DC\qiaomimi\model")


def describe_value(value):
    """简单描述保存对象中的值。"""
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device})"
        )

    if isinstance(value, dict):
        return f"dict，包含 {len(value)} 个键"

    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}，长度 {len(value)}"

    return f"{type(value).__name__}: {value!r}"


for path in sorted(model_dir.glob("*.pth")):
    print("\n" + "=" * 80)
    print(f"文件：{path.name}")
    print(f"大小：{path.stat().st_size:,} bytes")

    try:
        # weights_only=False 可以识别更多旧式保存格式；
        # 只检查自己训练、可信来源的文件。
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        print(f"顶层类型：{type(obj)}")

        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"键数量：{len(keys)}")

            for key in keys[:30]:
                print(f"  {key}: {describe_value(obj[key])}")

            if len(keys) > 30:
                print(f"  ... 另外还有 {len(keys) - 30} 个键")

            # 粗略判断保存格式
            if "model_state_dict" in obj:
                print("判断：训练检查点，包含 model_state_dict。")
            elif "state_dict" in obj:
                print("判断：检查点，模型权重可能位于 state_dict 中。")
            elif all(isinstance(v, torch.Tensor) for v in obj.values()):
                print("判断：很可能是直接保存的 model.state_dict()。")
            else:
                print("判断：字典格式，可能是检查点或自定义保存内容。")

        elif isinstance(obj, torch.nn.Module):
            print("判断：保存的是整个 PyTorch 模型对象。")
            print(obj)

        else:
            print("判断：保存的是其他 Python 对象。")
            print(repr(obj)[:1000])

    except Exception as exc:
        print(f"读取失败：{type(exc).__name__}: {exc}")
