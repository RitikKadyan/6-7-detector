# Six Seven Thing

This is a little Python script that watches your hands through your webcam and waits for you to do the legendary **“6–7” hand motion** (hands yeeting up/down in opposite directions).  
When you pull it off, a GIF shows up on the right side of your screen.  
That’s… basically the whole story.

---

## **What It Does**

- Tracks your hands using MediaPipe  
- Finds your palm center (works even if your palm faces up)  
- Watches how fast your hands move  
- Detects the “6–7” opposite-direction snap  
- Shows your webcam on the left  
- Slaps a GIF on the right  
- Draws motion trails because they look sick  
- Shows FPS  
- Logs important events only (no spam)

---

## **Requirements**

Minimal stuff:

```
pip install opencv-python mediapipe imageio numpy
```

Tkinter should already be included (unless your Python install is cursed).

Put your GIF in the same folder and call it:

```
67.gif
```

Or change the filename in the script if you like chaos.

---

## **How to Run**

```
python main.py
```

Hit `q` to exit.

---

## **How It Works**

- Pulls the last 5 palm-center positions  
- Measures how far they moved  
- If one hand shoots up while the other shoots down?  
  → Congratulations, you just “six-sevend”  
- It plays a GIF like a proud parent  

No deep learning, no magic — just velocity math and questionable ambition.

---

## **Performance Tips**

If FPS feels low:

- Lower your webcam resolution  
- Close Chrome tabs (especially the 37 you “need”)  
- Don’t run it on a toaster  
- The script already does optimizations, but it’s not Tony Stark tech  

---

## **Troubleshooting**

**Hand not detected?**  
Open your fingers more. MediaPipe hates T-rex arms.

**GIF looks cursed?**  
Don’t use a 4K GIF exported from Pixar.

**Laggy?**  
Your laptop might be “fine for school work” energy.

---

## **Extra Ideas**

- Add sound effects  
- Different GIF for each direction  
- Scoreboard  
- Twitch chat reacts (if you really want to go wild)

---

## **License**

Use it, tweak it, break it, fix it.
