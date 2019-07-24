import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


import com.github.javacliparser.FileOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.TimingUtils;
import moa.gui.visualization.DataPoint;
import moa.streams.clustering.FileStream;
import utils.InstanceKernel;
import utils.MicroCluster;

public class Run {

	double threshold = 0;
	int minClusters = 0;
	int kNumber = 0;
	String defaultfile = "";
	String outFile = "";
	int numInstan = 0;
	double pRand = 100;

	public Run(int kNumber, double threshold, String defaultfile, int numInstan, int minClusters, String outFile, double pRand) {

		this.kNumber = kNumber;
		this.threshold = threshold;
		this.minClusters = minClusters;

		this.defaultfile = defaultfile;
		this.outFile = outFile;
		this.numInstan = numInstan;
		this.pRand = pRand;
	}

	public void run() throws Exception {

		// value should be part of initialization
		double p_rand = pRand;
		
		BufferedWriter output = null;

		try {
			File file = new File("../example.txt");
			if (file.exists())
				file.delete();
			output = new BufferedWriter(new FileWriter(file));

		} catch (IOException e) {
			e.printStackTrace();
		}

		// datasets
		FileStream stream = new FileStream();
		stream.arffFileOption = new FileOption("arffFile", 'f', "ARFF file to load.", defaultfile, "arff", false);
		stream.prepareForUse();

		// classifier
		KNN_SWC learner = new KNN_SWC();
		// set number of k
		learner.kOption.setValue(kNumber);
		learner.setOutFile(outFile);
		learner.prepareForUse();
		learner.setModelContext(stream.getHeader());
		learner.setLimit_Option(numInstan);
		learner.setPRand(p_rand);
		
		int numberSamplesCorrect = 0;
		int numberSamples = 0;
		int testSamples = 0;

		long timestamp = -1;

		// values for evaluation
//		boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

		// true classification
//		ArrayList<ClassificationMeasures> measures = new ArrayList<>();

		ArrayList<String> classes = new ArrayList<>();

		// stream
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance().getData();
			DataPoint dp = new DataPoint(inst, 1);

			String classValue = String.valueOf(inst.classValue());

			// array of classes
			int pos = classes.indexOf(classValue);
			if (pos == -1)
				classes.add(classValue);

			InstanceKernel inKe = new InstanceKernel(inst, inst.numAttributes(), timestamp);

			// train
			if (numberSamples <= numInstan) {

				MicroCluster in = new MicroCluster(inKe, classValue, "normal", timestamp);
				learner.trainOnInstance(in, minClusters, dp);

			} else {

				if (timestamp == -1) {
					System.out.println("...Testing the model...");
					
					stream.restart();
					stream.prepareForUse();
					
					numberSamples = 0;
					testSamples = 0;
					while (stream.hasMoreInstances()) {
						timestamp += 1;
						
						inst = stream.nextInstance().getData();
						if (numberSamples <= numInstan) {
							testSamples++;
							if(learner.testOfflineModel(inst)) {
								numberSamplesCorrect++;
							}
							// time to merge closests clusters
							if (timestamp != 0 & timestamp % 1000 == 0) {
								System.out.println("numberSamplesCorrect " + numberSamplesCorrect);
								System.out.println("testSamples " + testSamples);
								double accuracy = 100.0 * (double) numberSamplesCorrect / (double) testSamples;
								double time = TimingUtils.nanoTimeToSeconds(
										TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
								System.out.println(numberSamples + " instances processed with " + accuracy
										+ "% accuracy in " + time + " seconds.");
								numberSamplesCorrect = 0;
								testSamples = 0;
							}
							
							numberSamples++;
							
						} else {
							break;
						}
						
					}
					numberSamplesCorrect = 0;
					timestamp=0;
					testSamples = 0;
					numberSamples = numInstan+1;
				}

				timestamp += 1;
				testSamples++;
				
				if(timestamp == 0) {
					System.out.println("...Real Testing...");
				}
				
				int answer = learner.classify(inKe, classes, timestamp);

				if (answer == 1) {
					numberSamplesCorrect++;
				}
				
				// remove unknown from accuracy
				if (answer == 2) {
					testSamples--;
				}

				// partial accuracy
				if (timestamp!= 0 & timestamp % 1000 == 0) {
					double accuracy = 100.0 * (double) numberSamplesCorrect / (double) testSamples;
					// calculating execution time
					double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
					System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");
					numberSamplesCorrect = 0;
					testSamples = 0;
				}

			}

			numberSamples++;
		}

		double accuracy = 100.0 * (double) numberSamplesCorrect / (double) testSamples;
		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
		System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");
		learner.stopFile();
		output.close();
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		int k = 1;
		double thres = 1.1;
		double pRand = 100;
		System.out.println(Integer.toString(k));
		
		File file = new File("results/");
		if (file.exists() && file.isDirectory())
			file.delete();
		else
			file.mkdir();
		
		Run r1 = new Run(k, thres, "datasets/SynD.arff", 25000, 100,		 "results/"+Integer.toString(k)+"KNN/synd/", pRand);
		Run r2 = new Run(k, thres, "datasets/GEARS_2C_2D.arff", 19000, 100,  "results/"+Integer.toString(k)+"KNN/gears/", pRand);
		Run r3 = new Run(k, thres, "datasets/moa3.arff", 10000, 100,   		 "results/"+Integer.toString(k)+"KNN/moa/", pRand);
		Run r4 = new Run(k, thres, "datasets/1CDT.arff", 1520, 100, 		 "results/"+Integer.toString(k)+"KNN/CDT/", pRand);
		Run r5 = new Run(k, thres, "datasets/covtypeNorm.arff", 58101, 100,  "results/"+Integer.toString(k)+"KNN/covertype/", pRand);
		Run r6 = new Run(k, thres, "datasets/kdd99.arff", 49402, 100,   	 "results/"+Integer.toString(k)+"KNN/kdd/", pRand);
		Run r7 = new Run(k, thres, "datasets/UG_2C_2D_on.arff", 5000, 100,   "results/"+Integer.toString(k)+"KNN/UG/",pRand);
		
		System.out.println("baseline experiments ");
		System.out.println("baseline synd ");
		r1.run();
		System.out.println("baseline gears ");
		r2.run();
		System.out.println("baseline moa ");
		r3.run();
		System.out.println("baseline cdt");
		r4.run();
		System.out.println("baseline covertype ");
		r5.run();
		//System.out.println("baseline kdd ");
		//r6.run();
		System.out.println("baseline UG ");
		r7.run();

	}

}
