from django.shortcuts import render

# Create the main view function

def main(request):

    # Get the content image from the user

    content_image_path = request.POST.get('content_image')

    content_image = tf.io.read_file(content_image_path)

    content_image = tf.image.decode_jpeg(content_image)

    content_image = tf.image.resize(content_image, (256, 256))

    content_image = tf.expand_dims(content_image, axis=0)

    # Get the style image from the user

    style_image_path = request.POST.get('style_image')

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

    # Return the stylized image to the user

    return render(request, 'main.html', {'stylized_image': stylized_image})
