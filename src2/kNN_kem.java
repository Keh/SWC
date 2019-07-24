
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

import java.awt.List;
import java.io.FileWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import utils.InstanceKernel;
import utils.NearestNeighbours;
import weka.core.pmml.jaxbbindings.MININGFUNCTION;
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
public class kNN_kem extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	private boolean initialized;

	public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 1000, 1,
			Integer.MAX_VALUE);

	public int limit_Option = 1000;

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

	ArrayList<InstanceKernel> windowKernel;
	SummClusters clusters;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.classes = new ArrayList<>();
			this.windowKernel = new ArrayList<>();
			this.window = new Instances(context, 0);
			this.window.setClassIndex(context.classIndex());

		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		this.windowKernel = null;
		this.initialized = false;
		this.classes = null;
		clusters = new SummClusters();
	}

	public void trainOnInstance(InstanceKernel inst) {

		timestamp++;
		InstanceKernel[] model = new InstanceKernel[100];
		InstanceKernel trueLabel;

		// window is empty -> initial training phase
		if (!initialized) {
			
			// amount of classes before test phase
			if (classes.indexOf(inst.getLabel()) == -1) {
				classes.add(inst.getLabel());
			}
			
			// first elements of the window
			if (windowKernel.size() < this.limit_Option) {
				inst.setType("normal");
				windowKernel.add(inst);

			} else {
				initialized = true;
//				System.out.println(windowKernel.size()); output importante
				// reduce prototypes
				model = clusters.trainOff(windowKernel);
				windowKernel.clear();

				// window with the prototypes
				windowKernel = new ArrayList<>(Arrays.asList(model));
//				
				System.out.println("timestamp " + timestamp + " " + windowKernel.get(0).getRadius());

				// test information
				for(int i = 0; i < model.length; i++) {
					System.out.println("id " + model[i].getId() );
					System.out.println("label " + model[i].getLabel() );
					System.out.println("id " + model[i].getType() );
					System.out.println("radius " + model[i].getRadius() );
				}
			}
		} else {
//			// an unknown instance arrive from the stream
//			inst.setType("unknown");
//			// get kOption neighbours 
//			ArrayList<NearestNeighbours> neighbours = kClosestNeighbor(inst, kOption.getValue());			
//		
//			// get classification label -> return double[] votes with amount of votes per label 
//			double [] votes = testInstance(inst, neighbours);
//			// inst.label = max(votes)
//
//			// train model with that instance
////			clusters.trainOn(inst);
//			windowKernel = clusters.getModelResult();
//
//			// remove instances - CUIDADO!!!!
////			if (this.limit_Option <= this.windowKernel.size()) {
////				this.windowKernel.remove(0);
////			}
//
//			// add online
////			this.windowKernel.add(inst);

		}

	}

	// get distance from k closest windowKernel
	public ArrayList<NearestNeighbours> kClosestNeighbor(InstanceKernel inst, int kOption) {

		InstanceKernel closestKernel = null;

		NearestNeighbours nN = null;
		NearestNeighbours[] votes = new NearestNeighbours[kOption];

		double minDistance = Double.MAX_VALUE;
//		Random rand = new Random();
		for (int i = 0; i < votes.length; i++) {
//			int n = rand.nextInt(this.windowKernel.size());
			double dis = distance(inst.getCenter(), this.windowKernel.get(i).getCenter());
			votes[i] = new NearestNeighbours(i, dis, this.windowKernel.get(i));
		}
		
		for (int i = 0; i < this.windowKernel.size(); i++) {
			double distance = distance(inst.getCenter(), this.windowKernel.get(i).getCenter());
			int first = 0;
			double maxDistance = Double.MIN_VALUE;
			for (int j = 0; j < votes.length; j++) {
				if (votes[j].getDistance() > maxDistance) {
					maxDistance = votes[j].getDistance();
					first = j;
				}
			}

			if (distance < votes[first].getDistance()) {
				closestKernel = this.windowKernel.get(i);
				nN = new NearestNeighbours(i, distance, closestKernel);
				votes[first] = nN;

			}

		}
		ArrayList<NearestNeighbours> votesList;
		votesList = new ArrayList<>(Arrays.asList(votes));
		return votesList;

//		return null;
	}

	// classification phase
	public double[] testInstance(InstanceKernel inst, ArrayList<NearestNeighbours> neighbours) {
		// neste momento - a distance da instance para os k vizinhos mais proximos.
		// por voto, obtem-se a quantidade para cada label de acordo com a distancia.
		double[] votes = new double[classes.size()];
//		System.out.println("instance label " + inst.getLabel());
		for(int i = 0; i < neighbours.size(); i++) {
			double foo = Double.parseDouble(neighbours.get(i).getKernel().getLabel());
//			System.out.println("radius " + neighbours.get(i).getKernel().getRadius());
//			System.out.println("center " + Arrays.toString(neighbours.get(i).getKernel().getCenter()));
			int foo2 =(int) foo;
			votes[foo2] ++;
//			System.out.println("distance " + neighbours.get(i).getDistance());
//			System.out.println("" + neighbours.get(i).getKernel().getLabel());
//			System.out.println("" + neighbours.get(i).getKernel().getType());
		}
		

		// highest vote
//		System.out.println(Arrays.toString(votes));
//		System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
		// unnecessary(?) begins
		// instead of sending only the closet instance, we send the first and the second.
		// why????
		double maxValue = Double.MIN_VALUE;
		double secondValue = Double.MIN_VALUE;
		
		int index = 99;
	    for (int i = 0; i < votes.length; i++) {
	        if (votes[i] > maxValue) {
	        	secondValue = maxValue;
	            maxValue = votes[i];
	            index = i;
	        } else if (maxValue > secondValue ) 
	        	secondValue = votes[i]; 
	        
	    }
//	    System.out.println("first Value " + maxValue);
//        System.out.println("second Value " + secondValue);
//	    System.out.println("index " + index);
		// unnecessary ends
	    
	    double[] realVotes = new double[2];
		return realVotes;
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

	private static double distance(double[] pointA, double[] pointB) {
		double distance = 0.0;
		for (int i = 0; i < pointA.length; i++) {
			double d = pointA[i] - pointB[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}
}