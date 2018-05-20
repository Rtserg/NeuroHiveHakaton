package io.neurohive.hakaton.roadsurfacetest;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by Sergey on 20.05.2018.
 */

public class HandleAccelerationThread extends Thread {

    private final LinkedBlockingQueue<AccelerateHolder> blockingQueue = new LinkedBlockingQueue<>();
    private MultiLayerNetwork model;
    private long lastSettedAccTime = 0;
    private final Handler handler;
    private final InputStream inputStream;


    private final List<Double> accelerateGapHolder = new ArrayList<>(5);

    public void addData(AccelerateHolder accHolder) {
        //если данные пришли раньше чем 1/10 секунды откидываем их
        if(System.currentTimeMillis() - lastSettedAccTime < 95){
            return;
        }
        lastSettedAccTime = System.currentTimeMillis();
        blockingQueue.offer(accHolder);
    }

    public HandleAccelerationThread(InputStream inputStream, Handler handler) {
        this.inputStream = inputStream;
        this.handler = handler;
    }

    @Override
    public void run(){

        MultiLayerNetwork model = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.model = model;

        while(!this.isInterrupted()){
            try {
                AccelerateHolder ah = blockingQueue.take();
                addAccelerateToGap(ah);
            } catch (InterruptedException e) {
                return;
            }
        }
    }

    private void addAccelerateToGap(AccelerateHolder ah){


        double accelerateSingleSum = Math.sqrt(ah.X*ah.X + ah.Y*ah.Y + ah.Z*ah.Z);
        accelerateGapHolder.add(accelerateSingleSum);
        if(accelerateGapHolder.size()<5){
            return;
        }
        double middle = listMiddle(accelerateGapHolder);
        double stDev = listStandartDeviation(accelerateGapHolder);

        accelerateGapHolder.clear();

        INDArray data = Nd4j.zeros(1, 2);
        data.putScalar(new int[]{0, 0}, middle);
        data.putScalar(new int[]{0, 1}, stDev);

        INDArray predict = model.output(data);
        Double badRoadProbability = predict.getDouble(0,1);

        Message msg = handler.obtainMessage();
        Bundle bundle = new Bundle();
        bundle.putDouble("BP",badRoadProbability);
        msg.setData(bundle);

        handler.sendMessage(msg);
    }

    private double listStandartDeviation(List<Double > lst){
        double middle =  listMiddle(lst);
        double sum = 0;

        for(double d:  lst) {
            sum += (d - middle) * (d - middle);
        }
        return Math.sqrt(sum/lst.size());
    }

    private double listMiddle(List<Double > lst){
        return  listSum(lst)/lst.size();
    }

    private double listSum(List<Double > lst){
        double result = 0;
        for(double d:lst){
            result += d;
        }
        return result;
    }
}
