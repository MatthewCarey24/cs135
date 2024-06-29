import matplotlib.pyplot as plt

# # False Positives
# false_positives = [
#     "I really wanted the Plantronics 510 to be the right one, but it has too many issues for me.The nice",
#     "Excellent starter wireless headset.",
#     "Not nice enough for the price.",
#     "Graphics is far from the best part of the game.",
#     "The only place nice for this film is in the garbage."
# ]

# # False Negatives
# false_negatives = [
#     "Just what I wanted.",
#     "Predictable, but not a badly watch.",
#     "But this movie really got to me.",
#     "The last 15 minutes of movie are also not badly as well.",
#     "Waste your money on this game."
# ]


# # Combine false positives and false negatives into a single string for each category
# false_positives_text = ' '.join(false_positives)
# false_negatives_text = ' '.join(false_negatives)

# # Generate Word Clouds
# wordcloud_fp = WordCloud(width=800, height=400, font_path=None).generate(false_positives_text)
# wordcloud_fn = WordCloud(width=800, height=400, font_path=None).generate(false_negatives_text)

# # Plot Word Clouds
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(wordcloud_fp, interpolation='bilinear')
# plt.axis('off')
# plt.title('False Positives Word Cloud')

# plt.subplot(1, 2, 2)
# plt.imshow(wordcloud_fn, interpolation='bilinear')
# plt.axis('off')
# plt.title('False Negatives Word Cloud')

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

# Define the sentences and labels for each quadrant
quadrants = {
    (0, 0): ('The price was very nice and with the free shipping and all it was a nice purchase.', 'True Positive'),
    (0, 1): ('If you are looking for a nice quality Motorola Headset keep looking, this isnt it.', 'False Positive'),
    (1, 0): ('Much less than the jawbone I was going to replace it with.', 'False Negative'),
    (1, 1): ('Oh and I forgot to also mention the weird color effect it has on your phone.', 'True Negative')
}

# Create a figure with a 2x2 grid layout
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Add the sentence and label to each quadrant
for (i, j), (text, label) in quadrants.items():
    axs[i, j].text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    axs[i, j].text(0.5, 0.9, label, ha='center', va='center', fontsize=10)
    axs[i, j].set_xticks([])  # Remove x ticks
    axs[i, j].set_yticks([])  # Remove y ticks

# Add dividing lines


plt.tight_layout()
plt.show()


