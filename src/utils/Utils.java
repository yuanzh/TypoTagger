package utils;

public class Utils {

	public static void Assert(boolean assertion) 
	{
		if (!assertion) {
			(new Exception()).printStackTrace();
			System.exit(1);
		}
	}
		
	public static double max(double[] vec) 
	{
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			max = Math.max(max, vec[i]);
		return max;
	}
	
	public static double min(double[] vec) 
	{
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			min = Math.min(min, vec[i]);
		return min;
	}
	
	public static double logSumExp(double x, double y) 
	{
		if (x == Double.NEGATIVE_INFINITY && x == y)
			return Double.NEGATIVE_INFINITY;
		else if (x < y)
			return y + Math.log1p(Math.exp(x-y));
		else 
			return x + Math.log1p(Math.exp(y-x));
	}
	
	public static double[] dot(double[]... vecs)
	{
		Utils.Assert(vecs.length > 0);
		int N = vecs[0].length;
		double[] ret = new double[N];
		for (int i = 0; i < N; ++i) {
			ret[i] = 1.0;
			for (double[] vec : vecs)
				ret[i] *= vec[i];
		}
		return ret;
	}
	
	public static double[] dot_s(double[] ret, double[]... vecs) {
		int N = ret.length;
		for (int i = 0; i < N; ++i) {
			double r = 1.0;
			for (double[] vec : vecs)
				r *= vec[i];
			ret[i] = r;
		}
		return ret;
	}
	
	public static double sum(double[] vec) {
		double sum = 0.0;
		for (int i = 0, L = vec.length; i < L; ++i)
			sum += vec[i];
		return sum;
	}
	
	public static double dotsum(double[] vec1, double[] vec2)
	{
		double sum = 0.0;
		for (int i = 0, N = vec1.length; i < N; ++i) {
			sum += vec1[i] * vec2[i];
		}
		return sum;
	}
	
}
