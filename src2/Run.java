import java.util.ArrayList;

import com.github.javacliparser.FileOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.TimingUtils;
import moa.streams.clustering.FileStream;
import utils.ClassificationMeasures;
import utils.InstanceKernel;

public class Run {


	public void run(int numInstances, boolean isTesting) {

		// datasets
		String defaultfile = "/home/kemilly/euler/moa3.arff";
		FileStream stream = new FileStream();
		stream.arffFileOption = new FileOption("arffFile", 'f', "ARFF file to load.", defaultfile, "arff", false);
		stream.prepareForUse();

		// classifier
		kNN_kem learner = new kNN_kem();
		// set number of k 
		learner.kOption.setValue(10);
		learner.prepareForUse();
		learner.setModelContext(stream.getHeader());
		System.out.println("test");

		int numberSamplesCorrect = 0;
		int numberSamples = 0;

		// values for evaluation
//		boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
//		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

		ArrayList<String> classes = new ArrayList<>();
		int number = 0;

		// stream
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance().getData();

			String classValue = String.valueOf(inst.classValue());
//			System.out.println(classValue);

			// array of classes
			int pos = classes.indexOf(classValue);
			if (pos == -1)
				classes.add(classValue);
			
			if(number <= numInstances ) {
				// Instance instance, int dimensions, long timestamp,String label, String type
				InstanceKernel in = new InstanceKernel(inst, inst.numAttributes(), 1, classValue , "normal");
				
				// index, for now, is for double values. However, some generalization should be necessary
				// same idea from original moa knn should be applied. vector with classes, each time a new class arrive, it is added to the vector.
				// that way, we always have a double set of classes.
				learner.trainOnInstance(in);
			}
			
			
			
			number++;
//			System.out.println("number " + number);

//			if (learner.correctlyClassifies(inst)) {
//				numberSamplesCorrect++;
//
//			}

		}

//		System.out.println(learner.window.size() + " number of classes " + classes.size());
//
//		// preparing second streaming
//		stream2.prepareForUse();
//
//		// learner.setModelContext(stream2.getHeader());
//
//		// test part
//		// true classification
//		ArrayList<ClassificationMeasures> measures = new ArrayList<>();
//		double newValue = 0;
//		while (stream2.hasMoreInstances()) {
//			Instance testInst = stream2.nextInstance().getData();
//			ClassificationMeasures cM = new ClassificationMeasures(String.valueOf(testInst.classValue()));
//			int truef = 0;
//
//			if (isTesting) {
//				// correct
//				double[] votes = learner.getVotesForInstance(testInst);
//				int vote = (int) Utils.maxIndex(votes);
//
//				newValue = vote;
//
//				if (learner.correctlyClassifies(testInst)) {
//					numberSamplesCorrect++;
//					// for compare
//					truef++;
//				}
//				int notNew = 0;
//				if (!measures.isEmpty()) {
//					for (int i = 0; i < measures.size(); i++) {
//						if (measures.get(i).getName().equals(cM.getName())) {
//							measures.get(i).setN(measures.get(i).getN() + 1);
//							if (truef != 0)
//								measures.get(i).setClassified(measures.get(i).getClassified() + 1);
//							notNew++;
//							// System.out.println("é igual " + cM.getName());
//						}
//					}
//					if (notNew == 0) {
//						if (truef != 0)
//							cM.setClassified(1);
//						measures.add(cM);
//					}
//				} else {
//					// System.out.println("só acontece uma vez " + cM.getName());
//					if (truef != 0)
//						cM.setClassified(1);
//					measures.add(cM);
//				}
//				// int random = min + (int) (Math.random() * ((max - min) + 1));
//				// if (random < 5) {
//				// // System.out.println("class valeu before " + testInst.classValue()
//				// // + "\n" + "new value " + newValue);
//				// testInst.setClassValue(newValue); // without true label
//				// learner.trainOnInstance(testInst);
//				// // System.out.println("class valeu after " + testInst.classValue());
//				// }
//			}
//			numberSamples++;
//		}
//
//		for (int i = 0; i < measures.size(); i++)
//			System.out.println(measures.get(i).getName() + " - " + measures.get(i).getN() + " - "
//					+ measures.get(i).getClassified());
//
//		// System.out.println(numberSamplesCorrect);
//		double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
//		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
//		System.out.println(
//				numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Run r = new Run();
		r.run(1000, true);

	}

}
