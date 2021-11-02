using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Drawing.Imaging;
using System.Collections.Generic;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;

namespace EMGUCV
{
    public partial class Form1 : Form
    {
        VideoCapture capture;
        static readonly CascadeClassifier cascadeClassifier = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        bool is_trained = false;
        bool is_snap = false;
        List<Mat> TrainedFaces = new List<Mat>();
        List<int> PersonLabel = new List<int>();
        EigenFaceRecognizer recognizer;
        Mat m=new Mat();
        public Form1()
        {
            InitializeComponent();
        }

        private void VideoCapture_ImageGrabbed(object sender, EventArgs e)
        {
            try
            {
                capture.Retrieve(m);
                var image = m.ToBitmap();

                //Face Detection
                Rectangle[] faces = cascadeClassifier.DetectMultiScale(m, 1.1, 4);
                foreach (var r in faces)
                {
                    using (Graphics graphics = Graphics.FromImage(image))
                    {
                        using (Pen pen = new Pen(Color.Red, 1))
                        {
                            graphics.DrawRectangle(pen, r);
                            //CvInvoke.PutText(m,"Face",new Point(10,r.Height-10),Emgu.CV.CvEnum.FontFace.HersheySimplex,1.0,new Bgr(Color.Red).MCvScalar);
                        }
                        pictureBox1.Image = image;
                    }

                    //Face Recognition
                    Image<Bgr, byte> resultImage = m.ToImage<Bgr, byte>();
                    resultImage.ROI = r;
                    pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;
                    if (pictureBox2.Image == null)
                    {
                        pictureBox2.Image = resultImage.ToBitmap();
                    }

                    //Snap Picture
                    if (is_snap)
                    {
                        var path = Directory.GetCurrentDirectory() + @"\Mike";
                        if (!Directory.Exists(path))
                        {
                            Directory.CreateDirectory(path);
                        }
                        for (int i = 0; i < 50; i++)
                        {
                            resultImage.Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic).Save(path + @"\Mike_" + i +".jpg");
                        }
                        MessageBox.Show("Snap Finished!");
                        is_snap = false;
                    }

                    if(is_trained)
                    {
                        Image<Gray, byte> grayFaceResult = resultImage.Convert<Gray,byte>().Resize(200,200,Emgu.CV.CvEnum.Inter.Cubic);
                        var result = recognizer.Predict(grayFaceResult);

                        CvInvoke.PutText(m, "Mike", new Point(r.X - 2, r.Y - 2),
                            FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                        CvInvoke.Rectangle(m, r, new Bgr(Color.Green).MCvScalar, 2);
                        pictureBox1.Image = m.ToBitmap();
                    }
                }
            }
            catch (Exception ex)
            {
                throw;
            }
        }

        private void startToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                capture.Start();
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }
        }

        private void stopToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (capture != null)
            {
                capture.Pause();
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            if (capture == null)
            {
                capture = new Emgu.CV.VideoCapture();
            }
            capture.Set(Emgu.CV.CvEnum.CapProp.FrameHeight, 1080);
            capture.Set(Emgu.CV.CvEnum.CapProp.FrameWidth, 1920);
            capture.ImageGrabbed += VideoCapture_ImageGrabbed;
            capture.Start();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void snapToolStripMenuItem_Click(object sender, EventArgs e)
        {
            is_snap = true;
        }

        private void trainToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int count = 0;
            int threshold = 7000;
            try
            {
                var path = Directory.GetCurrentDirectory() + @"\Mike";
                var files = Directory.GetFiles(path, "*.jpg",SearchOption.AllDirectories);
                foreach(var f in files)
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(f);
                    TrainedFaces.Add(trainedImage.Mat);
                    PersonLabel.Add(count);
                    count++;
                }
                recognizer = new EigenFaceRecognizer(count,threshold);
                recognizer.Train( new VectorOfMat(TrainedFaces.ToArray()), new VectorOfInt(PersonLabel.ToArray()));
                is_trained = true;
                MessageBox.Show("Model Trained!");
            }
            catch (Exception ex)
            {
                is_trained = false;
                throw;
            }
        }
    }
}
