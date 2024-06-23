package com.example.izquierdaorderecha

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.hardware.Camera
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.izquierdaorderecha.ml.ModelUnquant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback {
    private lateinit var surfaceView: SurfaceView
    private lateinit var imageView: ImageView
    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var btnPredict: Button
    private val labels = listOf("Izquierdo", "Derecho")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surfaceView)
        imageView = findViewById(R.id.imageView)
        val buttonTakePhoto: Button = findViewById(R.id.buttonTakePhoto)
        val buttonPredict: Button = findViewById(R.id.buttonPredict)
        val buttonUploadPhoto: Button = findViewById(R.id.buttonUploadPhoto)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        buttonTakePhoto.setOnClickListener { takePhoto() }
        buttonPredict.setOnClickListener { predict() }
        buttonUploadPhoto.setOnClickListener { openGallery() }

        val buttonDeletePhoto: Button = findViewById(R.id.buttonDeletePhoto)
        buttonDeletePhoto.setOnClickListener { deletePhoto() }

        btnPredict = findViewById(R.id.buttonPredict)
        btnPredict.setOnClickListener {
            if (imageView.drawable != null) {
                val drawable = imageView.drawable as BitmapDrawable
                val bitmap = drawable.bitmap
                predictImage(bitmap)
            } else {
                Toast.makeText(this, "No hay imagen para predecir", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startCamera() {
        val holder: SurfaceHolder = surfaceView.holder
        holder.addCallback(this)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        camera = Camera.open()
        try {
            camera?.setPreviewDisplay(holder)
            camera?.startPreview()
        } catch (e: IOException) {
            Log.e(TAG, "Error setting up preview display", e)
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (holder.surface == null) {
            return
        }

        try {
            camera?.stopPreview()
        } catch (e: Exception) {
            // Ignore: tried to stop a non-existent preview
        }

        try {
            camera?.setPreviewDisplay(holder)
            camera?.startPreview()
        } catch (e: IOException) {
            Log.e(TAG, "Error starting camera preview", e)
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        camera?.release()
        camera = null
    }

    private fun takePhoto() {
        camera?.takePicture(null, null, Camera.PictureCallback { data, _ ->
            val photoFile = File(
                externalMediaDirs.firstOrNull(),
                SimpleDateFormat(FILENAME_FORMAT, Locale.US)
                    .format(System.currentTimeMillis()) + ".jpg"
            )

            try {
                val fos = photoFile.outputStream()
                fos.write(data)
                fos.close()

                val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                val rotation = getRotationFromImage(Uri.fromFile(photoFile))
                val rotatedBitmap = rotateBitmap(bitmap, rotation)

                imageView.setImageBitmap(rotatedBitmap)
                imageView.visibility = ImageView.VISIBLE
                surfaceView.visibility = SurfaceView.GONE
            } catch (e: IOException) {
                Log.e(TAG, "Error saving photo", e)
            }
        })
    }

    private fun predict() {
        Toast.makeText(this, "Realizando predicción...", Toast.LENGTH_SHORT).show()
    }

    private fun predictImage(bitmap: Bitmap) {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        val processedImage = imageProcessor.process(tensorImage)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        val model = ModelUnquant.newInstance(this)

        val outputs = model.process(inputFeature0)
        val result = outputs.outputFeature0AsTensorBuffer

        val maxIndex = result.floatArray.indices.maxByOrNull { result.floatArray[it] } ?: -1
        val predictedClass = if (maxIndex != -1) labels[maxIndex] else "Unknown"

        val resultText = "Predicción: $predictedClass\nProbabilidades: ${result.floatArray.joinToString(", ")}"
        val txtPred = findViewById<TextView>(R.id.classPredicted)
        txtPred.text = resultText

        // Show result in a toast
        Toast.makeText(this, resultText, Toast.LENGTH_LONG).show()

        model.close()
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, REQUEST_CODE_SELECT_IMAGE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_SELECT_IMAGE && resultCode == Activity.RESULT_OK) {
            val selectedImageUri: Uri? = data?.data
            if (selectedImageUri != null) {
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
                imageView.setImageBitmap(bitmap)
                imageView.visibility = ImageView.VISIBLE
                surfaceView.visibility = SurfaceView.GONE
                // Llamar a la función de predicción aquí
                predictImage(bitmap)
            }
        }
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun getRotationFromImage(uri: Uri): Int {
        val exif = ExifInterface(uri.path!!)
        return when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
    }

    private fun rotateBitmap(source: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun deletePhoto() {
        AlertDialog.Builder(this)
            .setMessage("¿Estás seguro de que quieres borrar la foto?")
            .setPositiveButton("Sí") { _, _ ->
                deleteConfirmed()
            }
            .setNegativeButton("No", null)
            .show()
    }

    private fun deleteConfirmed() {
        imageView.visibility = ImageView.GONE
        surfaceView.visibility = SurfaceView.VISIBLE

        val photoFile = File(
            externalMediaDirs.firstOrNull(),
            SimpleDateFormat(FILENAME_FORMAT, Locale.US)
                .format(System.currentTimeMillis()) + ".jpg"
        )
        if (photoFile.exists()) {
            photoFile.delete()
            Toast.makeText(baseContext, "Photo deleted", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_CODE_SELECT_IMAGE = 11
    }
}