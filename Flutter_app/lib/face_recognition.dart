import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

class FaceRecognition extends StatefulWidget {
  const FaceRecognition({Key? key}) : super(key: key);

  @override
  _FaceRecognitionState createState() => _FaceRecognitionState();
}

class _FaceRecognitionState extends State<FaceRecognition> {
  late Interpreter _interpreter;
  late List<String> _classes;
  File? _selectedImage;
  String _predicted = "No prediction yet";

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadClasses();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter =
          await Interpreter.fromAsset("assets/final_FR_model_lite.tflite");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> _loadClasses() async {
    try {
      final classesJson = await rootBundle.loadString('assets/classes.json');
      setState(() {
        _classes = List<String>.from(jsonDecode(classesJson));
      });
    } catch (e) {
      print("Error loading classes: $e");
      setState(() {
        _classes = ["Unknown"];
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _predicted = "Predicting..."; // Reset prediction before running
      });
      _runModel(File(pickedFile.path));
    }
  }

  Future<void> _runModel(File imageFile) async {
    setState(() {
      _predicted = "Predicting...";
    });

    try {
      // Load and decode the image
      final image = img.decodeImage(await imageFile.readAsBytes());
      if (image == null) {
        setState(() {
          _predicted = "Invalid image selected!";
        });
        return;
      }

      // Resize the image to 224x224
      final resizedImage = img.copyResize(image, width: 224, height: 224);

      // Convert the image to a normalized float32 input tensor
      final inputBuffer = Float32List(224 * 224 * 3);
      for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
          final pixel = resizedImage.getPixel(j, i);
          final index = (i * 224 + j) * 3;
          inputBuffer[index] =
              img.getRed(pixel).toDouble(); // Normalize to [0, 1]
          inputBuffer[index + 1] = img.getGreen(pixel).toDouble();
          inputBuffer[index + 2] = img.getBlue(pixel).toDouble();
        }
      }

      // Validate input tensor shape
      final inputTensorShape = _interpreter.getInputTensor(0).shape;
      if (inputTensorShape.length != 4 ||
          inputTensorShape[1] != 224 ||
          inputTensorShape[2] != 224 ||
          inputTensorShape[3] != 3) {
        setState(() {
          _predicted = "Model input shape mismatch!";
        });
        return;
      }

      // Prepare the output buffer with batch dimension
      final outputTensorShape = _interpreter.getOutputTensor(0).shape;
      if (outputTensorShape.length != 2 || outputTensorShape[0] != 1) {
        setState(() {
          _predicted = "Model output shape mismatch!";
        });
        return;
      }
      final outputBuffer =
          Float32List(outputTensorShape[0] * outputTensorShape[1])
              .reshape([1, outputTensorShape[1]]);

      // Run inference
      _interpreter.run(inputBuffer.reshape([1, 224, 224, 3]), outputBuffer);

      // Extract predictions (flatten the batch dimension)
      final predictions = outputBuffer[0];
      final predictedIndex = predictions
          .indexOf(predictions.reduce((double a, double b) => a > b ? a : b));
      print("predictions: $predictions");
      print("Predicted index: $predictedIndex");
      // Update the UI with the predicted class
      setState(() {
        _predicted = _classes.isNotEmpty ? _classes[predictedIndex] : "Unknown";
      });
    } catch (e) {
      setState(() {
        _predicted = "Error during prediction: $e";
      });
      print("Error during prediction: $e");
    }
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.sizeOf(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Face Recognition App",
          style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
        ),
        backgroundColor: Color(0xFF003366),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Container(
              margin: const EdgeInsets.all(20),
              height: 250,
              width: size.width,
              child: DottedBorder(
                borderType: BorderType.RRect,
                radius: const Radius.circular(12),
                color: Colors.blueGrey,
                strokeWidth: 1,
                dashPattern: const [5, 5],
                child: SizedBox.expand(
                  child: FittedBox(
                      child: _selectedImage != null
                          ? Image.file(_selectedImage!, fit: BoxFit.cover)
                          : const Icon(
                              Icons.image_outlined,
                              color: Colors.blueGrey,
                            )),
                ),
              ),
            ),
            const SizedBox(height: 20),
            _predicted == "Predicting..."
                ? const CircularProgressIndicator(
                    // strokeWidth: 6.0,
                    color: Colors.blue,
                  )
                : Text(
                    "Predicted: $_predicted",
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.blueAccent,
                    ),
                  ),
            const SizedBox(height: 20),
            Padding(
              padding: const EdgeInsets.fromLTRB(40, 40, 40, 0),
              child: Material(
                elevation: 3,
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  width: size.width,
                  height: 50,
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Color(0xFF003366)),
                  child: Material(
                    borderRadius: BorderRadius.circular(20),
                    color: Colors.transparent,
                    child: InkWell(
                      splashColor: Colors.transparent,
                      highlightColor: Colors.transparent,
                      onTap: () {
                        _pickImage(ImageSource.camera);
                      },
                      child: const Center(
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.camera_alt, color: Colors.white),
                            SizedBox(width: 10),
                            Text(
                              'Camera',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(40, 40, 40, 20),
              child: Material(
                elevation: 3,
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  width: size.width,
                  height: 50,
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Color(0xFF003366)),
                  child: Material(
                    borderRadius: BorderRadius.circular(20),
                    color: Colors.transparent,
                    child: InkWell(
                      splashColor: Colors.transparent,
                      highlightColor: Colors.transparent,
                      onTap: () {
                        _pickImage(ImageSource.gallery);
                      },
                      child: const Center(
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.photo_library, color: Colors.white),
                            SizedBox(width: 10),
                            Text(
                              'Gallery',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
