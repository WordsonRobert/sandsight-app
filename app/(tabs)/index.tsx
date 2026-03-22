import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Button, Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

// --- TFJS POLYFILLS FOR KERAS 3 PREPROCESSING LAYERS ---
// These stop the "Unknown layer: Normalization" crash
class Normalization extends tf.layers.Layer {
  static className = 'Normalization';
  constructor(config: any) { super(config || {}); }
  computeOutputShape(inputShape: any) { return inputShape; }
  call(inputs: any) { return Array.isArray(inputs) ? inputs[0] : inputs; }
}
tf.serialization.registerClass(Normalization);

class Rescaling extends tf.layers.Layer {
  static className = 'Rescaling';
  constructor(config: any) { super(config || {}); }
  computeOutputShape(inputShape: any) { return inputShape; }
  call(inputs: any) { return Array.isArray(inputs) ? inputs[0] : inputs; }
}
tf.serialization.registerClass(Rescaling);
// ------------------------------------------------------

export default function Home() {
  const [permission, requestPermission] = useCameraPermissions();
  const [tfReady, setTfReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [results, setResults] = useState<string>('');
  
  const cameraRef = useRef<any>(null);
  const beachModelRef = useRef<tf.LayersModel | null>(null);
  const grainModelRef = useRef<tf.LayersModel | null>(null);

  useEffect(() => {
    async function prepare() {
      try {
        setResults("1. Waking up TF...");
        await tf.ready();
        
        // --- CHECK ENGINE (CRITICAL) ---
        const engine = tf.getBackend();
        if (engine === 'cpu') {
          setResults("❌ ERROR: Stuck on CPU! AI will freeze.");
          setTfReady(false);
          return; // Stop the code here so it doesn't freeze your phone
        }
        
        setResults(`Engine is ${engine} 🚀. Linking Beach...`);
        
        // --- LOAD BEACH (ISOLATION TEST) ---
        const beachJson = require('../../assets/images/models/beach/model.json');
        const beachWeights = [
          require('../../assets/images/models/beach/group1-shard1of5.bin'),
          require('../../assets/images/models/beach/group1-shard2of5.bin'),
          require('../../assets/images/models/beach/group1-shard3of5.bin'),
          require('../../assets/images/models/beach/group1-shard4of5.bin'),
          require('../../assets/images/models/beach/group1-shard5of5.bin'),
        ];
        
        setResults("3. Loading Beach Brain...");
        beachModelRef.current = await tf.loadLayersModel(bundleResourceIO(beachJson, beachWeights));

        // Note: We are entirely skipping the Grain model for this test!
        setResults("Beach Loaded! 🟢");
        setTfReady(true);
      } catch (err: any) {
        console.error("Model load error:", err);
        setResults("❌ CRASH: " + err.message);
        setTfReady(false);
      }
    }
    // DELAY LOADING: Give the camera 1.5 seconds to start before hogging the CPU with AI
    const timer = setTimeout(prepare, 1500); 
    return () => clearTimeout(timer);
  }, []);

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>SandSight needs your camera!</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  const analyzeSand = async () => {
    if (!cameraRef.current || !beachModelRef.current || !grainModelRef.current) return;
    setIsAnalyzing(true);
    setResults('Processing image...');

    try {
      const photo = await cameraRef.current.takePictureAsync();
      const resized = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 224, height: 224 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );
      setCapturedImage(resized.uri);

      setResults('Running AI Math...');
      // Using the fixed EncodingType import here
      const imgB64 = await FileSystem.readAsStringAsync(resized.uri, { 
        encoding: 'base64' as any 
      });
      const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);
      const imageTensor = decodeJpeg(raw);
      
      const normalizedTensor = imageTensor.div(255.0).expandDims(0);

      const beachPrediction = beachModelRef.current.predict(normalizedTensor) as tf.Tensor;
      const grainPrediction = grainModelRef.current.predict(normalizedTensor) as tf.Tensor;
      
      const beachData = await beachPrediction.data();
      const grainData = await grainPrediction.data();

      setResults(`Beach Type: ${beachData[0].toFixed(3)}\nGrain Size: ${grainData[0].toFixed(3)}`);
      setIsAnalyzing(false);

      tf.dispose([imageTensor, normalizedTensor, beachPrediction, grainPrediction]);
    } catch (error) {
      console.error(error);
      setResults('Error running AI.');
      setIsAnalyzing(false);
    }
  };

  return (
    <View style={styles.container}>
      {!capturedImage ? (
        <View style={styles.cameraContainer}>
          <CameraView style={styles.camera} facing="back" ref={cameraRef} />
          <View style={styles.overlay}>
            <View style={styles.header}>
              <Text style={styles.tfStatus}>
                TensorFlow: {tfReady ? "🟢 AI Loaded" : "🔴 Loading Brains..."}
              </Text>
              {/* This new line lets you see the steps while the camera is open! */}
              {!tfReady && results !== '' && (
                <Text style={styles.diagnosticText}>{results}</Text>
              )}
            </View>
            <View style={styles.buttonContainer}>
              <TouchableOpacity 
                style={[styles.button, !tfReady && styles.buttonDisabled]} 
                onPress={analyzeSand}
                disabled={!tfReady || isAnalyzing}
              >
                {isAnalyzing ? <ActivityIndicator color="white" /> : <Text style={styles.text}>📸 Analyze Sand</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      ) : (
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />
          <View style={styles.resultsBox}>
            <Text style={styles.resultsText}>{results}</Text>
          </View>
          <TouchableOpacity style={styles.button} onPress={() => setCapturedImage(null)}>
            <Text style={styles.text}>🔄 Scan Another Beach</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  cameraContainer: { flex: 1 },
  camera: { flex: 1 },
  overlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'space-between', zIndex: 10 },
  header: { paddingTop: 60, paddingHorizontal: 20, alignItems: 'center' },
  tfStatus: { backgroundColor: 'rgba(0,0,0,0.7)', color: 'white', padding: 12, borderRadius: 25, fontWeight: 'bold', overflow: 'hidden' },
  diagnosticText: { backgroundColor: 'rgba(0,0,0,0.7)', color: '#ffcc00', padding: 8, borderRadius: 10, marginTop: 10, fontWeight: 'bold', textAlign: 'center' },
  buttonContainer: { padding: 30, paddingBottom: 60 },
  button: { backgroundColor: '#006994', padding: 22, borderRadius: 18, alignItems: 'center', elevation: 5 },
  buttonDisabled: { backgroundColor: '#444' },
  text: { fontSize: 18, fontWeight: 'bold', color: 'white' },
  previewContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20, backgroundColor: '#111' },
  previewImage: { width: 280, height: 280, borderRadius: 20, marginBottom: 30, borderWidth: 3, borderColor: '#006994' },
  resultsBox: { backgroundColor: '#222', padding: 25, borderRadius: 15, marginBottom: 30, width: '100%', alignItems: 'center', borderWidth: 1, borderColor: '#333' },
  resultsText: { color: '#00ff00', fontSize: 20, fontWeight: 'bold', textAlign: 'center', lineHeight: 28 },
});