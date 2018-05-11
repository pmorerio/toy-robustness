# toy-robustness


video making
```
ffmpeg -framerate 15 -i image-%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
```

