
/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import java.util.Random;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import moa.gui.visualization.DataPoint;
import utils.ConDis;
import utils.DriftEvolution;
import utils.InstanceKernel;
import utils.MicroCluster;
import utils.NearestNeighbours;
import Clusters.ClusteringBla;
import Clusters.SummClusters;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;

/**
 * k Nearest Neighbor.
 * <p>
 *
 * Valid options are:
 * <p>
 *
 * -k number of neighbours <br>
 * -m max instances <br>
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class KNN_SWC extends AbstractClassifier implements MultiClassClassifier {
	private static final long serialVersionUID = 1L;

	private boolean initialized;
	private boolean newInitialized;

	public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 1000, 1,
			Integer.MAX_VALUE);

	public int limit_Option = 1000;

	protected double prob;

	public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption("nearestNeighbourSearch", 'n',
			"Nearest Neighbour Search to use", new String[] { "LinearNN", "KDTree" },
			new String[] { "Brute force search algorithm for nearest neighbour search. ",
					"KDTree search algorithm for nearest neighbour search" },
			0);

	ArrayList<String> classes;

	int C = 0;

	@Override
	public String getPurposeString() {
		return "kNN: special.";
	}

	protected Instances window;
	long timestamp = -1;

	double threshold = 1.1;
	double kernel = 1;
	double maxClusters = 0;
	double learningRate = 0.5;

	double p_rand = 20;

	String out1 = "";

	// this is just a test
	// double kernel

	ArrayList<MicroCluster> windowKernel;
	ArrayList<MicroCluster> neWindow;
	ArrayList<ConDis> conSeDist;
	ArrayList<DriftEvolution> driftList;

	SummClusters clusters;
	ClusteringBla clusteringBla;

	BufferedWriter output = null;
	BufferedWriter outputCenters = null;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.classes = new ArrayList<>();
			this.windowKernel = new ArrayList<>();
			this.neWindow = new ArrayList<>();
			this.conSeDist = new ArrayList<>();
			this.driftList = new ArrayList<>();
			this.window = new Instances(context, 0);
			this.window.setClassIndex(context.classIndex());

			clusteringBla = new ClusteringBla();

			try {
				File file = new File(out1 + "higia_output.txt");
				File file2 = new File(out1 + "higia_centers.txt");
				if (file.exists())
					file.delete();
				if (file2.exists())
					file2.delete();
				output = new BufferedWriter(new FileWriter(file));
				outputCenters = new BufferedWriter(new FileWriter(file2));
			} catch (IOException e) {
				e.printStackTrace();
			}

		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		this.windowKernel = null;
		this.neWindow = null;
		this.initialized = false;
		this.newInitialized = false;
		this.classes = null;
		clusters = new SummClusters();
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		if (inst.classValue() > C)
			C = (int) inst.classValue();

		if (this.window == null) {
			this.window = new Instances(inst.dataset());
		}
		// remove instances
		if (this.limitOption.getValue() <= this.window.size()) {
			this.window.delete(0);
		}
		this.window.add(inst);
	}

	public void setLimit_Option(int size) {
		this.limit_Option = size;
	}

	public void setPRand(double value) {
		this.p_rand = value;
	}

	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}

	public void trainOnInstance(MicroCluster inst, int minClusters, DataPoint dp) throws Exception {
		timestamp++;

		int lineSize = inst.getCenter().length - 1;
		double[] data = new double[lineSize];

		for (int j = 0; j < lineSize; j++) {
			data[j] = inst.getCenter()[j];
		}

		// window is empty -> initial training phase
		if (!initialized) {

			// amount of classes before test phase
			if (classes.indexOf(inst.getLabel()) == -1) {
				classes.add(inst.getLabel());
			}

			// first elements of the window
			if (this.windowKernel.size() < this.limit_Option) {
				this.windowKernel.add(inst);

			} else {

				initialized = true;
				// reduce prototypes
				ArrayList<MicroCluster> micros = clusteringBla.CluStream(this.windowKernel, minClusters, classes,
						timestamp, dp);
				this.windowKernel.clear();
				this.windowKernel = micros;
				this.maxClusters = this.windowKernel.size();
				System.out.println(maxClusters);
//				this.maxClusters = this.maxClusters * 1.2;
//				System.out.println(minClusters);
			}
		}

	}

	public boolean testOfflineModel(Instance inst) {

		int lineSize = inst.toDoubleArray().length - 1;
		double[] data = new double[lineSize];

		for (int j = 0; j < lineSize; j++) {
			data[j] = inst.toDoubleArray()[j];
		}

		Instance inst2 = new DenseInstance(1, data);

		InstanceKernel inKe = new InstanceKernel(inst2, inst2.numAttributes(), timestamp);

//		// get kOption neighbours 
		ArrayList<NearestNeighbours> neighbours = kClosestNeighbor(inKe, kOption.getValue());
		double[] votes = new double[classes.size()];
		double minDist = Double.MAX_VALUE;
		int index = 1;

		for (int i = 0; i < neighbours.size(); i++) {
			double foo = Double.parseDouble(neighbours.get(i).getKernel().getLabel());
			votes[(int) foo]++;
			index = neighbours.get(i).getIndex();
		}

		if ((double) max(votes) == inst.classValue()) {

			return true;
		}

		return false;
	}

	public void modelUpdate() throws IOException {

		// 3.2 Merge closest two kernels
		int closestA = 0;
		int closestB = 0;
		double minDistance = Double.MAX_VALUE;

		double radiusB = 0;
		double radiusA = 0;
		for (int i = 0; i < this.windowKernel.size(); i++) {
			double[] centerA = this.windowKernel.get(i).getCenter();
			radiusA = this.windowKernel.get(i).getRadius();
			for (int j = i + 1; j < this.windowKernel.size(); j++) {
				double dist = distance(centerA, this.windowKernel.get(j).getCenter());
				radiusB = this.windowKernel.get(j).getRadius();
				if (dist < minDistance) {
					minDistance = dist;
					closestA = i;
					closestB = j;
				}
			}

			assert (closestA != closestB);
			// heuristica
			if (minDistance <= (radiusA + radiusB)) {
//				System.out.println("acontece");
				this.windowKernel.get(closestA).add(this.windowKernel.get(closestB));
//				this.windowKernel.remove(closestB);
			}

		}
//		System.out.println(" " + this.windowKernel.size());

	}

	public void removeClusters() {

		for (int i = 0; i < this.windowKernel.size(); i++) {
			if (windowKernel.get(i).getTime() < (timestamp - 1000) && this.windowKernel.size() > 500) {
				windowKernel.remove(i);
				i--;
			}
		}

	}

	public int classify(InstanceKernel inst, ArrayList<String> claNormal, long time) throws Exception {
//		an unknown instance arrive from the stream
		timestamp = time;

		int lineSize = inst.getCenter().length - 1;
		double[] data = new double[lineSize];

		for (int j = 0; j < lineSize; j++) {
			data[j] = inst.getCenter()[j];
		}

		Instance inst2 = new DenseInstance(1, data);
		InstanceKernel inKe = new InstanceKernel(inst2, inst2.numAttributes(), timestamp);

//		// get kOption neighbours 
		ArrayList<NearestNeighbours> neighbours = kClosestNeighbor(inKe, kOption.getValue());

		String[] info = testInstance(inKe, neighbours, inst.getCenter()[lineSize]);

		output.write(Double.toString(inst.getCenter()[lineSize]));
		output.write(", " + info[0]);
		output.write(", " + info[1]);
		output.newLine();

		// write partial information of the model.
		if (timestamp % 1000 == 0) {
			for (int i = 0; i < this.windowKernel.size(); i++) {
//				System.out.println(this.windowKernel.get(i).getN());
				outputCenters.write(Integer.toString((int) timestamp));
				outputCenters.write(", ");
				for (int j = 0; j < lineSize; j++) {
					outputCenters.write(Double.toString(this.windowKernel.get(i).getCenter()[j]));
					outputCenters.write(", ");
				}
//				outputCenters.write(Arrays.toString(this.windowKernel.get(i).getCenter()));
//				outputCenters.write(", ");
				outputCenters.write(Double.toString(this.windowKernel.get(i).getRadius()));
				outputCenters.write(", ");
//				outputCenters.write(Double.toString(this.windowKernel.get(i).getThreshold()));

//				outputCenters.write(Double.toString(this.windowKernel.get(i).getId()));
//				outputCenters.write(", ");
				outputCenters.write(this.windowKernel.get(i).getLabel());
				outputCenters.newLine();
			}
		}

		// if true classification
		if (info[0].equals(Double.toString(inst.getCenter()[lineSize]))) {
			return 1;
		}

		// if unknown
		if (info[0].equals("")) {
			return 2;
		}

		return 0;
	}

	public void setOutFile(String string) {
		this.out1 = string;
	}

	// get distance from k closest windowKernel
	public ArrayList<NearestNeighbours> kClosestNeighbor(InstanceKernel inst, int kOption) {

		MicroCluster closestKernel = null;

		NearestNeighbours nN = null;
		NearestNeighbours[] votes = new NearestNeighbours[kOption];

		Random rand = new Random();

		for (int i = 0; i < votes.length; i++) {
			int n = rand.nextInt(this.windowKernel.size());
//			int n = i;
//			System.out.println(Arrays.toString(windowKernel.get(n).getCenter()));
			double dis = distance(inst.getCenter(), this.windowKernel.get(n).getCenter());
//			System.out.println("1 - " + dis);
			votes[i] = new NearestNeighbours(n, dis, this.windowKernel.get(n));
		}
		for (int i = 0; i < this.windowKernel.size(); i++) {
			double distance = distance(inst.getCenter(), this.windowKernel.get(i).getCenter());
//			System.out.println("indice " + i + " 2 - " + distance);
			int first = -1;
			double maxDistance = Double.MIN_VALUE;
			// biggest distance
			for (int j = 0; j < votes.length; j++) {
				if (votes[j].getDistance() > maxDistance) {
					maxDistance = votes[j].getDistance();
					first = j;
				}
			}

//			System.out.println("i from loop " + i);
			// replace with the biggest from the array votes
			if (distance < votes[first].getDistance()) {
//				System.out.println("i from if " + i);
				closestKernel = this.windowKernel.get(i);
				nN = new NearestNeighbours(i, distance, closestKernel);
				votes[first] = nN;
			}

		}

		ArrayList<NearestNeighbours> votesList;
		votesList = new ArrayList<>(Arrays.asList(votes));
//		System.out.println("$$$$$$$$$$$$$$$$$$$");
//		for(int i = 0; i < votesList.size(); i ++) {
//			System.out.println(Arrays.toString(votesList.get(i).getKernel().getCenter()));
//			System.out.println("list " + votesList.get(i).getIndex());
//			System.out.println("list " + votesList.get(i).getKernel().getLabel());
//		}

		return votesList;
	}

	// classification phase
	public String[] testInstance(InstanceKernel inst, ArrayList<NearestNeighbours> neighbours, double realLabel)
			throws Exception {

		// por voto, obtem-se a quantidade para cada label de acordo com a distancia.
		double[] votes = new double[classes.size()];
		String[] info = new String[2];
		info[0] = "";
		info[1] = "";
		double minDist = Double.MAX_VALUE;

		// 20% of chance in training
		int max = 100;
		int min = 0;
		int random = min + (int) (Math.random() * ((max - min) + 1));

		int index = 0;

		// real classification
		for (int i = 0; i < neighbours.size(); i++) {

			double foo = Double.parseDouble(neighbours.get(i).getKernel().getLabel());
//			// if the dist is less or equal to the radius of the closest cluster
			votes[(int) foo]++;

			// minimal dist to the closets cluster
			double dist = distance(inst.getCenter(), neighbours.get(i).getKernel().getCenter());

			if (dist < minDist) {
				minDist = dist;
				index = neighbours.get(i).getIndex();
			}

		}

		// majority vote
		int valor = (kOption.getValue() / 2) + 1;
		double thre = this.windowKernel.get(index).getThreshold();

		if (votes[max(votes)] >= valor) {
			// update closets cluster
			if (minDist <= (this.windowKernel.get(index).getRadius())) {

				info[0] = Double.toString(max(votes));
				info[1] = this.windowKernel.get(index).getType();
				if (random <= this.p_rand) {
					this.windowKernel.get(index).insert(inst.getCenter(), (int) timestamp);
				}
				return info;

			} else {
				// insert the instance as a prototype if it is in the uncertain border
				if (minDist <= (this.windowKernel.get(index).getRadius() * thre)) {

					info[0] = Double.toString(max(votes));
					info[1] = this.windowKernel.get(index).getType();
					long threshold = timestamp - 1000; // Kernels before this can be forgotten

//					3.1 Try to forget old kernels
					// 500 is the maximum size, but this should be a parameter
					if (random <= this.p_rand) {
						if (this.windowKernel.size() >= 500) {
							for (int i = 0; i < this.windowKernel.size(); i++) {
								if (this.windowKernel.get(i).getRelevanceStamp() < threshold) {
									MicroCluster element = new MicroCluster(inst, info[0], "extension", timestamp);
									this.windowKernel.set(i, element);
									return info;

								}
							}

						} else {
							double radius = minDist;
							MicroCluster element = new MicroCluster(inst, info[0], "extension", timestamp);
							this.windowKernel.add(element);
							return info;
						}
					}

				}

			}
		}

		return info;

	}

	public double[] getVotesForInstance(InstanceKernel inst) {

		Instance in = new DenseInstance((Instance) inst);

		double v[] = new double[C + 1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex() == 0) {
				search = new LinearNNSearch(this.window);
			} else {
				search = new KDTree();
				search.setInstances(this.window);
			}
			if (this.window.numInstances() > 0) {
				Instances neighbours = search.kNearestNeighbours(in,
						Math.min(kOption.getValue(), this.window.numInstances()));
				for (int i = 0; i < neighbours.numInstances(); i++) {
					v[(int) neighbours.instance(i).classValue()]++;
				}
			}
		} catch (Exception e) {
			return new double[classes.size()];
		}
		return v;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {

		double v[] = new double[C + 1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex() == 0) {
				search = new LinearNNSearch(this.window);
			} else {
				search = new KDTree();
				search.setInstances(this.window);
			}
			if (this.window.numInstances() > 0) {
				Instances neighbours = search.kNearestNeighbours(inst,
						Math.min(kOption.getValue(), this.window.numInstances()));
				for (int i = 0; i < neighbours.numInstances(); i++) {
					v[(int) neighbours.instance(i).classValue()]++;
				}
			}
		} catch (Exception e) {
			return new double[inst.numClasses()];
		}
		return v;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return false;
	}

	// max value
	public int max(double[] L) {
		double max = L[0];
		int index = 0;
		for (int i = 0; i < L.length; i++) {
			if (L[i] > max) {
				max = L[i];
				index = i;
			}
		}
		return index;
	}

	// min value
	public double min(double[] L) {
		double min = L[0];
		for (int i = 0; i < L.length; i++)
			if (L[i] < min)
				min = L[i];
		return min;
	}

	private static double distance(double[] pointA, double[] pointB) {
		double distance = 0.0;
		for (int i = 0; i < pointA.length; i++) {
			double d = pointA[i] - pointB[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}

	public void stopFile() throws IOException {
		output.close();
		outputCenters.close();
	}
}