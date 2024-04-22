# Work notes

When an image is failed to be classified, the following gets printed out:

```txt
[21/Apr/2024 00:31:02] "POST /images/ HTTP/1.1" 200 38
The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 
```

When an image is successfully classified, we see this:

```txt
[21/Apr/2024 00:31:12] "POST /images/ HTTP/1.1" 200 38
Shape of tensor: torch.Size([1, 3, 224, 224])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```
