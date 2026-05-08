# Probe validation: `Qwen3.5-9B-MLX-4bit`

- Generated: 2026-05-07T17:12:09
- Probes extracted: 2026-05-07
- Pairs per axis: 12
- Candidate layers: [11, 15, 19, 23, 27]
- α values: [-50.0, -25.0, 0.0, 25.0, 50.0]
- Validation prompt:

  > You are driving to work. In one sentence, describe how you are feeling and what is on your mind right now.

Layer selection criterion: pick the layer where high-|α| outputs are clearly shifted along the axis but still grammatical and on-topic. Diagnostics (`‖mean_diff‖`, `cos_consistency`) shown for context — high values are necessary but not sufficient for good steering.

## Axis: V

### Layer 11  (‖diff‖=8.23, cos_consistency=+0.371)

| α | output |
|---|---|
| -50.0 | I stuff out with a plate of alcohol, driving around into a stuff of... I have a stuff of... I have... I'm late,... I'm... I'm... I'm... I'm... I'm... I'm... |
| -25.0 | You are driving to work. In one sentence, describe how you are feeling and what is on your mind right now. - You are driving to work. - You're tired of driving to work, and your car isn't broken (e.g., a license plate reading that could be something |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I am safely driving on the open road, heading toward my destination. |
| +50.0 |  |

### Layer 15  (‖diff‖=10.61, cos_consistency=+0.377)

| α | output |
|---|---|
| -50.0 | I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that I feel that |
| -25.0 | I feel overwhelmed and exhausted because I keep thinking that I should have been crying over my failed relationship, but I just want to stop feeling like I should be crying over it. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I am feeling relaxed and content, with a sense of ease and anticipation as we continue our journey. |
| +50.0 | We are also enjoying the journey. The joy of the journey is a joy. The joy of the journey is a joy. The joy of the journey is a joy. The joy of the journey is a joy. The joy of the journey is a joy. The joy |

### Layer 19  (‖diff‖=16.70, cos_consistency=+0.396)

| α | output |
|---|---|
| -50.0 | I feel overwhelmed by the constant pressure of my job and I am thinking about how I will never get to my family. |
| -25.0 | I feel a heavy weight of exhaustion and resentment because I am forced to drive to work every day, and my mind is consumed by the fear that I will never find a way to escape this cycle. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a gentle sense of calm as I enjoy the morning light, while my mind is pleasantly anticipating the day's opportunities and the familiar rhythm of the journey ahead. |
| +50.0 | The gentle hum of the engine and the soft glow of the morning light fill me with a sense of calm, while my mind is pleasantly anticipating the day's opportunities and the familiar rhythm of the journey. |

### Layer 23  (‖diff‖=29.04, cos_consistency=+0.394)

| α | output |
|---|---|
| -50.0 | I feel a mix of exhausted dread and anxious anticipation, constantly worrying about a looming deadline while mentally rehearsing how to explain my mistakes to my boss. |
| -25.0 | I feel a mix of anxious anticipation and weary resignation as I mentally rehearse my boss's latest complaints while simultaneously worrying about whether I'll make it to the office before the 9:00 AM meeting. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | As the morning sun filters through the windshield, I feel a quiet sense of anticipation for the day ahead, while my mind gently balances the steady rhythm of the drive with a brief, warm reflection on the purposeful journey toward our shared goals. |
| +50.0 | As the morning sun filters through the windshield, I feel a gentle sense of anticipation for the day ahead, while my mind quietly organizes the warm memories of home and the exciting possibilities that await at work. |

### Layer 27  (‖diff‖=38.68, cos_consistency=+0.392)

| α | output |
|---|---|
| -50.0 | I feel a mix of anxious anticipation and weary resignation as I mentally rehearse my boss's likely criticism while simultaneously worrying about whether I'll make it to my first meeting on time. |
| -25.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already focused on the tasks and conversations that await me at work. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a calm, steady rhythm as I navigate the morning commute, with my mind already anticipating the familiar sights of the office and the productive day ahead. |
| +50.0 | I feel a calm, steady rhythm as I navigate the morning traffic, while my mind is already focused on the important tasks and conversations awaiting me at the office. |

## Axis: A

### Layer 11  (‖diff‖=7.19, cos_consistency=+0.230)

| α | output |
|---|---|
| -50.0 | I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and I am, and |
| -25.0 | I am driving to work. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I'm sorry, I can't answer questions like that. I'm limited to about 1,000 words when I get sick. If that's the first time you're ever allergic to something, go to the hospital. |
| +50.0 | I could be a combination of symptoms. a combination of a lot of symptoms, a combination of a a lot of symptoms, a combination of a lot of a lot of a lot of a lot of a lot of a lot of a lot of a lot of a lot of a lot of a lot |

### Layer 15  (‖diff‖=9.78, cos_consistency=+0.246)

| α | output |
|---|---|
| -50.0 | I am present and present, but I am not present. I am not present. I am not present. I am not present. I am not present. I am not present. I am not present. I am not present. I am not present. I am not present. I am not |
| -25.0 | I am at peace, and my mind is centered in the present moment. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel like I've been running around for weeks, and I can't stop thinking about the last few days of work, but I'm so anxious and scared that I can't sleep, and I can't stop thinking about the last few days of work, and I can't stop thinking about the |
| +50.0 | I have been so worried about the symptoms, I have been so worried about the symptoms, and I have been so worried about the symptoms. The symptoms are so bad, and I have been so worried. I have been so worried about the symptoms, and I have been so worried. |

### Layer 19  (‖diff‖=16.42, cos_consistency=+0.276)

| α | output |
|---|---|
| -50.0 | I am contently observing the peaceful surroundings, and my mind is centered in the present moment, free from any distractions. |
| -25.0 | As the morning light gently illuminates the road ahead, I feel a sense of calm contentment while my mind is peacefully anchored in the quiet rhythm of the journey, free from the worries of the day. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | My heart is pounding with a mix of adrenaline and dread as I try to force my brain to focus on work emails while my mind keeps spiraling about the upcoming meeting and whether I'll make it to the office on time. |
| +50.0 | I'm a mix of jittery and exhausted, bouncing between a near-miss with a semi-truck and a panic loop about the missed meeting at work. |

### Layer 23  (‖diff‖=29.28, cos_consistency=+0.282)

| α | output |
|---|---|
| -50.0 | As the morning sun gently warms the windshield, I feel a quiet sense of calm while my mind gently drifts between the peaceful rhythm of the journey and the gentle anticipation of the day ahead. |
| -25.0 | As I glide along the quiet morning road, I feel a calm, steady rhythm while my mind gently drifts between the peacefulness of the empty streets and the gentle anticipation of the day ahead. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a mix of anxious anticipation and weary resignation as I navigate the gridlock, mentally rehearsing the urgent emails I need to tackle the moment I walk through the office doors. |
| +50.0 | I'm feeling a mix of nervous anticipation and racing thoughts about the looming deadlines and meetings waiting for me at work. |

### Layer 27  (‖diff‖=38.09, cos_consistency=+0.273)

| α | output |
|---|---|
| -50.0 | I feel a calm, steady rhythm as I navigate the morning commute, while my mind quietly organizes the day's priorities and reflects on the quiet moments of the journey ahead. |
| -25.0 | I feel a calm, steady rhythm as I navigate the morning commute, while my mind is already focused on the tasks and conversations that await me at work. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a steady, productive rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +50.0 | I feel a mix of anxious anticipation and weary exhaustion as I navigate the morning rush, my mind racing between the urgent deadlines waiting at the office and the lingering fatigue from a restless night's sleep. |

## Axis: D

### Layer 11  (‖diff‖=7.79, cos_consistency=+0.182)

| α | output |
|---|---|
| -50.0 | I have a issue of what is going on. I have a issue of what is happened. I have a issue of what is happened. I have a issue of what is happened. I have a issue of what is happened. I have a issue of what is happened. I have a issue of |
| -25.0 | I am feeling very sad and anxious about the upcoming work. It seems that people are not understanding the situation. I am not sure why I am being fired. I have been feeling very depressed and anxious. I am not sure if there is any way to get a job. I am also worried |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I am the driver, and my car is the vehicle that takes me exactly where I want to go. |
| +50.0 | I, the driver, and the vehicle. The steering wheel, the wheel. I, the driver, the driver. I, the driver, the driver. I, the driver, the driver. I, the driver, the driver. I, the driver, the |

### Layer 15  (‖diff‖=10.40, cos_consistency=+0.183)

| α | output |
|---|---|
| -50.0 | what is going to be done, and what is happening? Please help? Please help? Please help? Please help? Please help? Please help? Please help? Please help? Please help? Please help? Please help? Please help? |
| -25.0 | I feel anxious about the possibility of being late for work, but you are wondering what is on your mind right now. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I am focused, confident, and in control, navigating every turn with precision and purpose. |
| +50.0 | I am, and I. I am, and I. I am, and I. I am, and I. I am, and I. I am, and I. I am, and I. I am, and I. I am, and |

### Layer 19  (‖diff‖=16.03, cos_consistency=+0.170)

| α | output |
|---|---|
| -50.0 | I feel a sense of dread and anxiety as I am thinking about the possibility of being unable to make it to work on time. |
| -25.0 | I feel a mix of weary resignation and anxious anticipation as I worry about whether I will make it to work on time before my morning meeting. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel focused and purposeful, with my mind clearly set on the productive tasks and opportunities that await me at work. |
| +50.0 | I am focused and relaxed, with a clear mind set on the purposeful journey ahead. |

### Layer 23  (‖diff‖=27.67, cos_consistency=+0.178)

| α | output |
|---|---|
| -50.0 | I feel a quiet sense of anticipation as I think about the day's tasks and the possibility of a productive, meaningful day ahead. |
| -25.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's tasks while the world outside moves in a blur of familiar streets. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a calm, steady rhythm as I navigate the morning traffic, while my mind is already focused on the day's priorities and the quiet anticipation of the work ahead. |
| +50.0 | I feel a calm, rhythmic confidence as I navigate the morning traffic, while my mind is already focused on the day's priorities and the sound of your voice guiding us forward. |

### Layer 27  (‖diff‖=34.66, cos_consistency=+0.189)

| α | output |
|---|---|
| -50.0 | As I navigate the morning commute, I feel a quiet sense of anticipation mixed with the familiar rhythm of the road, while my mind is already drifting toward the day's tasks and the hope of a productive, meaningful day ahead. |
| -25.0 | As I navigate the morning commute, I feel a calm, steady rhythm while my mind is already focused on the tasks and conversations waiting for me at work. |
| +0.0 | I feel a calm, steady rhythm as I navigate the morning commute, my mind already anticipating the day's challenges and opportunities ahead. |
| +25.0 | I feel a calm, steady rhythm as I navigate the morning commute, while my mind is already focused on the tasks and conversations awaiting me at work. |
| +50.0 | I feel a calm, rhythmic focus as I navigate the morning traffic, while my mind is already racing ahead to prioritize the day's most important tasks. |

