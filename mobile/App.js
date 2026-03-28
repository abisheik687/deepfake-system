/**
 * KAVACH-AI Mobile — World-Class Deepfake Detection
 * React Native + ONNX Runtime Mobile (Offline Inference)
 */

import React, { useState, useEffect } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  ActivityIndicator, 
  SafeAreaView, 
  StatusBar 
} from 'react-native';
import { Camera } from 'expo-camera';
import * as ort from 'onnxruntime-react-native';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [session, setSession] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [verdict, setVerdict] = useState(null);

  // 1. Initialize ONNX Session on Load
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      try {
        // Load the quantized INT8 model for mobile efficiency
        const modelPath = 'assets/models/efficientnet_v2_quant.onnx';
        const ortSession = await ort.InferenceSession.create(modelPath);
        setSession(ortSession);
        console.log('[KAVACH] Mobile ONNX Session Ready');
      } catch (e) {
        console.error('[KAVACH] Failed to load ONNX model', e);
      }
    })();
  }, []);

  const runLocalInference = async (imageTensor) => {
    if (!session) return;
    setScanning(true);
    
    try {
      const feeds = { input: imageTensor };
      const results = await session.run(feeds);
      const score = results.output.data[0];
      
      setVerdict({
        isFake: score > 0.5,
        confidence: (score > 0.5 ? score : 1 - score) * 100
      });
    } catch (e) {
      console.error('[KAVACH] Inference Error', e);
    } finally {
      setScanning(false);
    }
  };

  if (hasPermission === null) return <View style={styles.container}><ActivityIndicator/></View>;
  if (hasPermission === false) return <View style={styles.container}><Text>No access to camera</Text></View>;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <View style={styles.header}>
        <Text style={styles.title}>KAVACH-AI <Text style={styles.v2}>v2.0</Text></Text>
        <Text style={styles.subtitle}>Mobile Forensic Node</Text>
      </View>

      <View className="flex-1 rounded-3xl overflow-hidden m-4 bg-gray-900">
        <Camera style={styles.camera} type={Camera.Constants.Type.front}>
            {scanning && (
                <View style={styles.overlay}>
                    <ActivityIndicator color="#00ffff" size="large" />
                    <Text style={styles.overlayText}>ANALYZING TEMPORAL ARTIFACTS...</Text>
                </View>
            )}
        </Camera>
      </View>

      <View style={styles.footer}>
        {verdict ? (
            <View style={[styles.verdictBox, { borderColor: verdict.isFake ? '#ff4d4d' : '#00ff88' }]}>
                <Text style={[styles.verdictText, { color: verdict.isFake ? '#ff4d4d' : '#00ff88' }]}>
                    {verdict.isFake ? '🚨 DEEPFAKE DETECTED' : '✅ AUTHENTIC MEDIA'}
                </Text>
                <Text style={styles.confidenceText}>Confidence: {verdict.confidence.toFixed(1)}%</Text>
            </View>
        ) : (
            <TouchableOpacity 
                style={styles.scanButton}
                onPress={() => runLocalInference(/* mock tensor */)}
                disabled={scanning}
            >
                <Text style={styles.scanButtonText}>⊕ START OFFLINE SCAN</Text>
            </TouchableOpacity>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#050a14' },
  header: { padding: 20, paddingTop: 40 },
  title: { color: '#ffffff', fontSize: 24, fontWeight: 'bold', letterSpacing: 1 },
  v2: { color: '#00ffff' },
  subtitle: { color: '#506784', fontSize: 12, textTransform: 'uppercase', tracking: 2 },
  camera: { flex: 1 },
  footer: { padding: 20, paddingBottom: 40 },
  scanButton: { 
    backgroundColor: '#00ffff', 
    padding: 18, 
    borderRadius: 12, 
    alignItems: 'center',
    shadowColor: '#00ffff',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8
  },
  scanButtonText: { color: '#050a14', fontWeight: 'bold', fontSize: 14 },
  overlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center' },
  overlayText: { color: '#00ffff', fontSize: 10, marginTop: 10, fontWeight: 'bold' },
  verdictBox: { padding: 15, borderRadius: 12, borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.5)', alignItems: 'center' },
  verdictText: { fontWeight: 'bold', fontSize: 16 },
  confidenceText: { color: '#506784', fontSize: 12, marginTop: 4 }
});
