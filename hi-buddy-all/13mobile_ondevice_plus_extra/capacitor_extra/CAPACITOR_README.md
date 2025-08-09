Capacitor Build Helpers
-----------------------

After you've built your web assets (either the www/ or the React build), copy them into the capacitor project and run:

npx cap copy
npx cap open android
npx cap open ios

Make sure AndroidManifest and Info.plist include camera and microphone permissions:
- Android: <uses-permission android:name="android.permission.RECORD_AUDIO"/>
           <uses-permission android:name="android.permission.CAMERA"/>
- iOS: NSCameraUsageDescription, NSMicrophoneUsageDescription in Info.plist
