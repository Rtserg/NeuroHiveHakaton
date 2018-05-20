package io.neurohive.hakaton;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

//тестирование сети
public class LoadNet {

    private static final String pathToLoadNet = "D:\\hak\\data\\ml2jData\\road_surface_test.zip";
    private static final String pathToTestInputFile = "D:\\hak\\data\\ml2jData\\test_input";
    private static final String pathToTestOutputFile = "D:\\hak\\data\\ml2jData\\test_output";

    public static void main(String[] args) throws IOException {
        File locationToLoad = new File(pathToLoadNet);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);
        System.out.println(testNet(restored));
    }


    private static int testNet(MultiLayerNetwork net) throws IOException {
        int lineCount = countLineInFile(pathToTestInputFile);
        int wrongPredict = 0;
        INDArray input = readInputFile(pathToTestInputFile);
        INDArray output = readInputFile(pathToTestOutputFile);

        INDArray predict = net.output(input);
        System.out.println(predict);
        for(int i = 0; i < lineCount; i++) {
            int outputValue = (int)Math.round(output.getDouble(i,0));
            int predictValue = (int)Math.round(predict.getDouble(i,0));
            if(outputValue != predictValue){
                wrongPredict++;
                System.out.println("Wrong predict number:" + (i+1));
            }
        }
        System.out.println("Percent:" + (double)wrongPredict*100/(double)lineCount);
        return wrongPredict;
    }

    private static INDArray readInputFile(String fileName) throws IOException {
        int lineCount = countLineInFile(fileName);
        INDArray data = Nd4j.zeros(lineCount, 2);
        BufferedReader buf=new BufferedReader(new FileReader(fileName));

        for(int i = 0; i < lineCount; i++) {
            String line = buf.readLine();
            String[] splitLine = line.split(";");
            double first = Double.parseDouble(splitLine[0]);
            double second = Double.parseDouble(splitLine[1]);
            data.putScalar(new int[]{i, 0}, first);
            data.putScalar(new int[]{i, 1}, second);
        }
        return data;
    }

    private static INDArray readOutputFile(String fileName) throws IOException {
        int lineCount = countLineInFile(fileName);
        INDArray data = Nd4j.zeros(lineCount, 2);
        BufferedReader buf=new BufferedReader(new FileReader(fileName));

        for(int i = 0; i < lineCount; i++) {
            String line = buf.readLine();
            String[] splitLine = line.split(";");
            int first = Integer.parseInt(splitLine[0]);
            int second = Integer.parseInt(splitLine[1]);
            data.putScalar(new int[]{i, 0}, first);
            data.putScalar(new int[]{i, 1}, second);
        }
        return data;
    }

    private static int countLineInFile(String fileName) throws IOException {
        int lineCount=0;
        BufferedReader buf=new BufferedReader(new FileReader(fileName));
        while((buf.readLine())!=null)
        {
            lineCount++;
        }
        return lineCount;
    }

}
