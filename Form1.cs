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
        static readonly CascadeClassifier eyeClassifier = new CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");
        static readonly CascadeClassifier faceClassifier = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        bool is_trained = false;
        bool is_snap = false;
        List<Mat> TrainedFaces = new List<Mat>();
        List<int> PersonLabel = new List<int>();
        List<string> PersonName = new List<string>();
        EigenFaceRecognizer recognizer;
        Mat m = new Mat();
        int picCount = 0;
        string username = "";
        public Form1()
        {
            InitializeComponent();
        }

        private void VideoCapture_ImageGrabbed(object sender, EventArgs e)
        {
            try
            {
                //Get from Webcam
                capture.Retrieve(m);
                Image<Bgr, byte> resultImage = m.ToImage<Bgr, byte>();

                //Face Detection Rectangle
                Rectangle[] faces = faceClassifier.DetectMultiScale(m, 1.3, 4);
                Rectangle[] eyes = eyeClassifier.DetectMultiScale(m, 1.3, 4);
                foreach (var r in faces)
                {
                    resultImage.ROI = r;

                    //Snap Picture
                    if (is_snap)
                    {
                        var path = Directory.GetCurrentDirectory() + @"\Mike";
                        if (!Directory.Exists(path))
                        {
                            Directory.CreateDirectory(path);
                        }
                        resultImage.Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic).Save(path + @"\" + textBox1.Text + "_" + picCount + ".jpg");
                        picCount++;
                    }

                    //If loaded train model
                    if (is_trained)
                    {
                        Image<Gray, byte> grayFaceResult = resultImage.Convert<Gray, byte>().Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic);
                        var result = recognizer.Predict(grayFaceResult);

                        if (result.Label < 0)
                        {
                            CvInvoke.Rectangle(m, r, new Bgr(Color.Red).MCvScalar, 2);
                        }
                        else
                        {
                            CvInvoke.PutText(m, PersonName[result.Label].ToString(), new Point(r.X - 2, r.Y - 2),
                                FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                            CvInvoke.Rectangle(m, r, new Bgr(Color.Green).MCvScalar, 2);
                        }
                    }
                    // Not using trained model
                    else
                    {
                        CvInvoke.Rectangle(m, r, new Bgr(Color.Red).MCvScalar, 1);
                    }
                }
                foreach (var y in eyes)
                {
                    CvInvoke.Rectangle(m, y, new Bgr(Color.Blue).MCvScalar, 2);
                }
                pictureBox1.Image = m.ToBitmap();
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
            if (!is_snap)
            {
                is_snap = true;
            }
            else
            {
                is_snap = false;
                MessageBox.Show("Finished Snapping...");
            }
        }

        private void trainToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int count = 0;
            double threshold = 7000;
            try
            {
                var path = Directory.GetCurrentDirectory() + @"\Mike";
                var files = Directory.GetFiles(path, "*.jpg", SearchOption.AllDirectories);
                foreach (var f in files)
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(f);
                    TrainedFaces.Add(trainedImage.Mat);
                    PersonLabel.Add(count);
                    int charLocation = Path.GetFileName(f).IndexOf('_');
                    PersonName.Add(Path.GetFileName(f).Substring(0, charLocation));
                    count++;
                }
                recognizer = new EigenFaceRecognizer(count, threshold);
                recognizer.Train(new VectorOfMat(TrainedFaces.ToArray()), new VectorOfInt(PersonLabel.ToArray()));
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
