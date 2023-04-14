import tkinter as tk

# Create the main window

root = tk.Tk()

# Create a label for the content image

content_image_label = tk.Label(root, text="Content Image")

content_image_label.pack()

# Create a canvas for the content image

content_image_canvas = tk.Canvas(root, width=256, height=256)

content_image_canvas.pack()

# Create a label for the style image

style_image_label = tk.Label(root, text="Style Image")

style_image_label.pack()

# Create a canvas for the style image

style_image_canvas = tk.Canvas(root, width=256, height=256)

style_image_canvas.pack()

# Create a button to generate the stylized image

generate_stylized_image_button = tk.Button(root, text="Generate Stylized Image", command=generate_stylized_image)

generate_stylized_image_button.pack()

# Create a label for the stylized image

stylized_image_label = tk.Label(root, text="Stylized Image")

stylized_image_label.pack()

# Create a canvas for the stylized image

stylized_image_canvas = tk.Canvas(root, width=256, height=256)

stylized_image_canvas.pack()
# Define a function to generate the stylized image

def generate_stylized_image():

    # Get the content image from the user

    content_image_path = input("Enter the path to the content image: ")

    content_image = tf.io.read_file(content_image_path)

    content_image = tf.image.decode_jpeg(content_image)

    content_image = tf.image.resize(content_image, (256, 256))

    content_image = tf.expand_dims(content_image, axis=0)

    # Get the style image from the user

    style_image_path = input("Enter the path to the style image: ")

    style_image = tf.io.read_file(style_image_path)

    style_image = tf.image.decode_jpeg(style_image)

    style_image = tf.image.resize(style_image, (256, 256))

    style_image = tf.expand_dims(style_image, axis=0)

    # Train the model

    model.fit(content_image, style_image, epochs=10, steps_per_epoch=100, verbose=1)

    # Generate the stylized image

    stylized_image = model.predict(content_image)

    # Save the stylized image

    tf.io.write_file('stylized_image.jpg', stylized_image)

    # Display the stylized image

    plt.imshow(stylized_image)

    plt.show()

# Start the main loop

root.mainloop()
# Exit the program

root.destroy()
