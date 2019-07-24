package Clusters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import utils.InstanceKernel;

public class SummClusters {

	private int timeWindow;
	private long timestamp = -1;
	private InstanceKernel[] kernels;
	private boolean initialized;
	private List<InstanceKernel> buffer; // Buffer for initialization with kNN
	private int bufferSize;
	private double t;
	private int m;

	private ArrayList<InstanceKernel> intModel; 
	
	public SummClusters() {

		this.timestamp = 1000; // user
		this.kernels = new InstanceKernel[100]; // user
		this.buffer = new LinkedList<InstanceKernel>();
		this.bufferSize = 100; // user
		this.t = 2; // user
		this.m = 100; // user
	}

	// distance between two points
	private static double distance(double[] pointA, double[] pointB) {
		double distance = 0.0;
		for (int i = 0; i < pointA.length; i++) {
			double d = pointA[i] - pointB[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}

	// calculating centers
	private static SphereCluster calculateCenter(ArrayList<Cluster> cluster, int dimensions) {
		double[] res = new double[dimensions];
		for (int i = 0; i < res.length; i++) {
			res[i] = 0.0;
		}

		if (cluster.size() == 0) {
			return new SphereCluster(res, 0.0);
		}else {
//			System.out.println("cluster.size() " + cluster.size()); // output importante
		}
		

		for (Cluster point : cluster) {
			double[] center = point.getCenter();
			for (int i = 0; i < res.length; i++) {
				res[i] += center[i];
			}
		}

		// Normalize
		for (int i = 0; i < res.length; i++) {
			res[i] /= cluster.size();
		}
		

		// Calculate radius
		double radius = 0.0;
		for (Cluster point : cluster) {
			double dist = distance(res, point.getCenter());
			if (dist > radius) {
				radius = dist;
			}
		}
		
		SphereCluster sc = new SphereCluster(res, radius);
		sc.setWeight(cluster.size());
		return sc;
	}

	// executing kMeans
	public static Clustering kMeans(int k, Cluster[] centers, List<? extends Cluster> data) {
		assert (centers.length == k);
		assert (k > 0);

		int dimensions = centers[0].getCenter().length;

		ArrayList<ArrayList<Cluster>> clustering = new ArrayList<ArrayList<Cluster>>();
		for (int i = 0; i < k; i++) {
			clustering.add(new ArrayList<Cluster>());
		}

		int repetitions = 100;
		while (repetitions-- >= 0) {
			// Assign points to clusters
			for (Cluster point : data) {
				double minDistance = distance(point.getCenter(), centers[0].getCenter());
				int closestCluster = 0;
				for (int i = 1; i < k; i++) {
					double distance = distance(point.getCenter(), centers[i].getCenter());
					if (distance < minDistance) {
						closestCluster = i;
						minDistance = distance;
					}
				}
				clustering.get(closestCluster).add(point);
			}

			// Calculate new centers and clear clustering lists
			SphereCluster[] newCenters = new SphereCluster[centers.length];
			for (int i = 0; i < k; i++) {
				newCenters[i] = calculateCenter(clustering.get(i), dimensions);
				clustering.get(i).clear();
				
			}
			centers = newCenters;
		}

		return new Clustering(centers);
	}

	// prototype reduction -> clustering cluStream
	public InstanceKernel[] trainOff(ArrayList<InstanceKernel> windowKernel) {
		// 0. Initialize
		// for each instance
		for (int i = 0; i < windowKernel.size(); i++) {
			
			Instance instance = new DenseInstance(1, windowKernel.get(i).getCenter());

			int dim = instance.numValues();
			timestamp++;
			
			if (buffer.size() < bufferSize) {
				buffer.add(new InstanceKernel(instance, dim, timestamp, t, m, windowKernel.get(i).getLabel(), "normal"));
			}
		}
		
		int k = kernels.length;

		assert (k <= bufferSize);

		InstanceKernel[] centers = new InstanceKernel[k];
		
		//random
		Random rand = new Random();
		
		for (int j = 0; j < k; j++) {
			int n = rand.nextInt(buffer.size());			
			centers[j] = buffer.get(n); 
		}
		
		// constrain by label -> missing that part
		Clustering kmeans_clustering = kMeans(k, centers, buffer);
		
//		System.out.println("@@@@@@@@@@@ " + Arrays.toString(kernels));
		// passing the label and using the tag "normal"
		for (int j = 0; j < kmeans_clustering.size(); j++) {
			System.out.println("@@@@@@@@@@@ " + Arrays.toString(kmeans_clustering.get(j).getCenter()));
			Instance newInst = new DenseInstance(1.0, kmeans_clustering.get(j).getCenter());		
			
			kernels[j] = new InstanceKernel(newInst, kmeans_clustering.get(j).getCenter().length, timestamp, t, m,
					centers[j].getLabel(), "normal");

			System.out.println(Arrays.toString(kmeans_clustering.get(j).getCenter()));
			System.out.println(Arrays.toString(kernels[j].getCenter()));
			
//			kernels[j] = new InstanceKernel(newInst, kmeans_clustering.get(j).getCenter().length, timestamp, t, m,
//					centers[j].getLabel(), "normal");

			
		}
		
		this.intModel = new ArrayList<>(Arrays.asList(kernels));
		
		buffer.clear();
		return kernels;
		
	}
	
	
	public void trainOn(InstanceKernel instKernel) {
		
		Instance instance = new DenseInstance(1, instKernel.getCenter());
		
		// 1. Determine closest kernel
		// kNN 
		InstanceKernel closestKernel = null;
		double minDistance = Double.MAX_VALUE;
		for ( int i = 0; i < this.intModel.size(); i++ ) {
			//System.out.println(i+" "+kernels[i].getWeight()+" "+kernels[i].getDeviation());
			double distance = distance(instance.toDoubleArray(), this.intModel.get(i).getCenter() );
			if ( distance < minDistance ) {
				closestKernel = this.intModel.get(i);
				minDistance = distance;
			}
		}

		// 2. Check whether instance fits into closestKernel
		double radius = 0.0;
		if ( closestKernel.getWeight() == 1 ) {
			// Special case: estimate radius by determining the distance to the
			// next closest cluster
			radius = Double.MAX_VALUE;
			double[] center = closestKernel.getCenter();
			for ( int i = 0; i < this.intModel.size(); i++ ) {
				if ( this.intModel.get(i) == closestKernel ) {
					continue;
				}
				
				double distance = distance(this.intModel.get(i).getCenter(), center );
				radius = Math.min( distance, radius );
			}
		} else {
			radius = closestKernel.getRadius();
		}
		
//		float threshold = (float) (radius * 1.1);
		
		// insert in the closest existing kernel -> isto funciona mesmo???
		if ( minDistance < radius ) {
			// Date fits, put into kernel and be happy
			closestKernel.insert( instance, timestamp );
			
			if(closestKernel.getType().equals("outlier")) {					
				// to turn a novelty
				if(closestKernel.getN() > 10) {
//					System.out.println("not outlier");
					closestKernel.setType("Novelty");				
				}
				
			}	
			
//			System.out.println(closestKernel.getType());
			
//			System.out.println("not outlier");
//			System.out.println(instKernel.getType());
//			System.out.println(instKernel.getLabel());
			return; //-> is this necessary?
		} else {
			// insert alone, in the dark
			instKernel.setType("outlier");
			this.intModel.add(instKernel);

			
		}		
		// HOW TO DECIDE THAT? 3.2 Merge closest two kernels
		int closestA = 0;
		int closestB = 0;
		minDistance = Double.MAX_VALUE;
		for ( int i = 0; i < this.intModel.size(); i++ ) {
			double[] centerA = this.intModel.get(i).getCenter();
			for ( int j = i + 1; j < this.intModel.size(); j++ ) {
				double dist = distance( centerA, this.intModel.get(j).getCenter() );
				if ( dist < minDistance ) {
					minDistance = dist;
					closestA = i;
					closestB = j;
				}
			}
		}
		assert (closestA != closestB);	
		this.intModel.get(closestA).add( this.intModel.get(closestB) );
		this.intModel.remove(closestB);
		
//		if(minDistance < (this.intModel.get(closestA).getRadius() + this.intModel.get(closestB).getRadius())) {
//			if(this.intModel.get(closestA).getLabel().equals(this.intModel.get(closestB).getLabel())) {
//				if(this.intModel.get(closestA).getType().equals(this.intModel.get(closestB).getType())) {
//					this.intModel.get(closestA).add( this.intModel.get(closestB) );
//					this.intModel.remove(closestB);
//				}
//				
////				System.out.println("@@@@@@@@@@@");
//			}
//		}
		
		
		
//		System.out.println(this.intModel.size());

	}
	
	public ArrayList<InstanceKernel> getModelResult() {

		return this.intModel;
	}

	// return list of microClustering
	public Clustering getMicroClusteringResult() {

		InstanceKernel[] res = new InstanceKernel[kernels.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = new InstanceKernel(kernels[i], t, m, "" , "");
		}

		return new Clustering(res);
	}

}
