package com.example.demo.controller;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import nu.pattern.OpenCV;

public class WebcamFaceDetectionExample extends JPanel {
    private BufferedImage image;
    private CascadeClassifier faceCascade;
    private VideoCapture videoCapture;

    public WebcamFaceDetectionExample() {
       OpenCV.loadLocally();

        // Load the face cascade XML file
        faceCascade = new CascadeClassifier();
        faceCascade.load("haarcascade_frontalface_default.xml");

        // Create a video capture object for the webcam
        videoCapture = new VideoCapture(0);

        // Create a frame to display the webcam feed
        JFrame frame = new JFrame("Webcam Face Detection");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setSize(640, 480);
        frame.setContentPane(this);
        frame.setVisible(true);

        // Start the webcam face detection loop
        while (true) {
            // Read a frame from the webcam
            Mat frameMat = new Mat();
            videoCapture.read(frameMat);

            // Detect faces in the frame
            MatOfRect faceDetections = new MatOfRect();
            faceCascade.detectMultiScale(frameMat, faceDetections);

            // Draw rectangles around the detected faces
            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frameMat, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            }

            // Convert the frame to a buffered image for display
            image = matToBufferedImage(frameMat);

            // Repaint the panel to update the displayed image
            repaint();
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image != null) {
            g.drawImage(image, 0, 0, null);
        }
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();
        byte[] sourceData = new byte[width * height * channels];
        mat.get(0, 0, sourceData);

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        final byte[] targetData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(sourceData, 0, targetData, 0, sourceData.length);

        return image;
    }

    public static void main(String[] args) {
        new WebcamFaceDetectionExample();
    }
}
