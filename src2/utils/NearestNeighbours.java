package utils;

public class NearestNeighbours {
	
	int index;
	double distance;
	InstanceKernel kernel;
	
	public NearestNeighbours(int i, double minDistance, InstanceKernel ik) {
		this.index = i;
		this.distance = minDistance;
		this.kernel = ik;
	}

	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public InstanceKernel getKernel() {
		return kernel;
	}

	public void setKernel(InstanceKernel kernel) {
		this.kernel = kernel;
	}
	
	
	

}
