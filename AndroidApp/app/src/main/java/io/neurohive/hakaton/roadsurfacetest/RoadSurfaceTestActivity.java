package io.neurohive.hakaton.roadsurfacetest;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

public class RoadSurfaceTestActivity extends AppCompatActivity implements SensorEventListener{
    private TextView message;
    private HandleAccelerationThread handleAccelerationThread;
    private View view;

    private SensorManager mSensorManager;
    private Sensor mSensor;

    @SuppressLint("HandlerLeak")
    private final Handler HANDLER = new Handler(){
        //если плохая дорога, то она секунду остается плохой
        private long waitTime = 0;
        @Override
        public void handleMessage(Message msg) {

            if(System.currentTimeMillis() - waitTime < 1000){
                return;
            }

            Bundle bd = msg.getData();
            double badProb = bd.getDouble("BP");
            if(badProb>0.7){
                message.setText("BAD ROAD");
                view.setBackgroundColor(Color.RED);
                waitTime = System.currentTimeMillis();
            }else{
                message.setText("GOOD ROAD");
                view.setBackgroundColor(Color.WHITE);
            }

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_road_surface_test);
        message = (TextView)findViewById(R.id.tvMessage);

        InputStream inputStream = getResources().openRawResource(R.raw.road_surface_test);

        handleAccelerationThread = new HandleAccelerationThread(inputStream,HANDLER);

        view = this.getWindow().getDecorView();

        handleAccelerationThread.setDaemon(true);
        handleAccelerationThread.start();

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            float[] values = event.values;
            // Movement
            float x = values[0];
            float y = values[1];
            float z = values[2];
            handleAccelerationThread.addData(new AccelerateHolder(x,y,z));
        }

    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }


    @Override
    protected void onResume() {
        super.onResume();
        // регистрация листенера
        mSensorManager.registerListener(this,
                mSensor,
                SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    protected void onPause() {
        // удаление листенера при выходе из прложения
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

}
