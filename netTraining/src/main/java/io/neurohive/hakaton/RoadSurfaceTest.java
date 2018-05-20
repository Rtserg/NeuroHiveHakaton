package io.neurohive.hakaton;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Чтение данных и обучение нейронной сети, для определения качества поверхности
 * дорожного покрытия.
 */
public class RoadSurfaceTest {
    private static final String pathToInputFile = "D:\\hak\\data\\ml2jData\\input";
    private static final String pathToOutputFile = "D:\\hak\\data\\ml2jData\\output";

    private static final String pathToTestInputFile = "D:\\hak\\data\\ml2jData\\test_input";
    private static final String pathToTestOutputFile = "D:\\hak\\data\\ml2jData\\test_output";

    //путь до файла сохранения сети
    private static final String pathToSaveNet = "D:\\hak\\data\\ml2jData\\road_surface_test.zip";

    private static final int NumberOfEpoch = 10000;

    public static void main(String[] args) throws IOException {

        INDArray inputs = readInputFile(pathToInputFile);
        INDArray outputs = readOutputFile(pathToOutputFile);

         DataSet ds = new DataSet(inputs, outputs);

        // создание конфигурации сети
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(new Sgd(0.1));
        //перемешать данные
        ds.shuffle();

        builder.biasInit(0);

        //создание сети
        ListBuilder listBuilder = builder.list();
        //создание полносвязного первого слоя
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        //два входа: полусекундное среднее значение и среднеквадратичное отклонение
        hiddenLayerBuilder.nIn(2);
        //10 нейронов первый скрытй слой
        hiddenLayerBuilder.nOut(10);
        //ф-ция активации сигмойд
        hiddenLayerBuilder.activation(Activation.SIGMOID);

        //задание начальных весов
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        //добавление слоя
        listBuilder.layer(0, hiddenLayerBuilder.build());

        //задание ф-ции потер выходного слоя
        Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
        //количество входов, должно соответвовать количеству выходов предыдущего слоя
        outputLayerBuilder.nIn(10);

        //создание выходного слоя
        outputLayerBuilder.nOut(2);
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        listBuilder.layer(1, outputLayerBuilder.build());


        listBuilder.pretrain(false);
        listBuilder.backprop(true);

        //создание и инициализия сети
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // отображение процесса обучения
        net.setListeners(new ScoreIterationListener(100));

        //обучение сети
        for( int i=0; i<NumberOfEpoch; i++ ) {
            net.fit(ds);
        }

        //тестирование и вывод результатов тестирования
        System.out.println(testNet(net));

        //сохранение сети на диск, для использования в приложении
        File locationToSave = new File(pathToSaveNet);
        ModelSerializer.writeModel(net, locationToSave, false);
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
