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
using Emgu.CV.Dnn;

namespace EMGUCV
{
    public partial class Form1 : Form
    {
        VideoCapture capture;
        static readonly CascadeClassifier eyeClassifier = new CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");
        static readonly CascadeClassifier faceClassifier = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        bool is_trained = false;
        bool is_snap = false;
        bool is_hand_landmark = false;
        List<Mat> TrainedFaces = new List<Mat>();
        List<int> PersonLabel = new List<int>();
        List<string> PersonName = new List<string>();
        EigenFaceRecognizer recognizer;
        Mat m = new Mat(400, 400, DepthType.Cv64F, 1);
        int picCount = 0;
        string username = "";
        static string prototxt = @"C:\openpose\models\hand\pose_deploy.prototxt";
        static string modelpath = @"C:\openpose\models\hand\pose_iter_102000.caffemodel";
        Net net = DnnInvoke.ReadNetFromCaffe(prototxt, modelpath);

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

                        if (result.Label < 0 || result.Label >= PersonLabel.Count)
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
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
                pictureBox1.Image = m.ToBitmap();

                //Hand Landmarking
                if(is_hand_landmark)
                {
                    //var hand = new Image<Bgr, Byte>(Path.Combine(Directory.GetCurrentDirectory(), "hand.jpg"));
                    var hand = m;
                    var blob = DnnInvoke.BlobFromImage(hand, 1.0 / 255.0, new Size(400, 400), new MCvScalar(0, 0, 0));


                    net.SetInput(blob);
                    net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);

                    var output = net.Forward();

                    var H = output.SizeOfDimension[2];
                    var W = output.SizeOfDimension[3];

                    var probMap = output.GetData();

                    int nPoints = 22;
                    int[,] POSE_PAIRS = new int[,] { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 0, 5 }, { 5, 6 }, { 6, 7 },
                        { 7, 8 }, { 0, 9 }, { 9, 10 }, { 10, 11 }, { 11, 12 }, { 0, 13 }, { 13, 14 }, { 14, 15 }, { 15, 16 },
                        { 0, 17 }, { 17, 18 }, { 18, 19 }, { 19, 20 } };

                    var points = new List<Point>();

                    for (int i = 0; i < nPoints; i++)
                    {
                        Matrix<float> matrix = new Matrix<float>(H, W);
                        for (int row = 0; row < H; row++)
                        {
                            for (int col = 0; col < W; col++)
                            {
                                matrix[row, col] = (float)probMap.GetValue(0, i, row, col);
                            }
                        }

                        double minVal = 0, maxVal = 0;
                        Point minLoc = default, maxLoc = default;
                        CvInvoke.MinMaxLoc(matrix, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                        var x = (hand.Width * maxLoc.X) / W;
                        var y = (hand.Height * maxLoc.Y) / H;

                        var p = new Point(x, y);
                        points.Add(p);
                        CvInvoke.Circle(hand, p, 5, new MCvScalar(0, 255, 0), -1);
                        CvInvoke.PutText(hand, i.ToString(), p, FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                    }

                    for (int i = 0; i < POSE_PAIRS.GetLongLength(0); i++)
                    {
                        var startIndex = POSE_PAIRS[i, 0];
                        var endIndex = POSE_PAIRS[i, 1];

                        if (points.Contains(points[startIndex]) && points.Contains(points[endIndex]))
                        {
                            CvInvoke.Line(hand, points[startIndex], points[endIndex], new MCvScalar(255, 0, 0), 2);
                        }
                    }
                    pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
                    pictureBox1.Image = hand.ToBitmap();
                }
            }
            catch (Exception ex)
            {
                throw;
            }
        }

        private void startToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (!is_hand_landmark)
            {
                is_hand_landmark = true;
            }
            else
            {
                is_hand_landmark = false;
                MessageBox.Show("Stopped Hand Lanmarking...");
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
                capture = new Emgu.CV.VideoCapture(0,VideoCapture.API.DShow);
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

        private void loadToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                var path = Directory.GetCurrentDirectory() + @"\model.efr";
                recognizer = new EigenFaceRecognizer(0, 7000);
                recognizer.Read(path);
                MessageBox.Show("Model Loaded!");
                is_trained = true;
            }
            catch (Exception ex)
            {
                is_trained = false;
                throw;
            }
        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                var path = Directory.GetCurrentDirectory() + @"\model.efr";
                recognizer.Write(path);
                var namepath = Directory.GetCurrentDirectory() + @"\model_name.efr";
                var labelpath = Directory.GetCurrentDirectory() + @"\model_label.efr";
                MessageBox.Show("Model Saved!");
            }
            catch (Exception ex)
            {
                is_trained = false;
                throw;
            }
        }
    }
}
