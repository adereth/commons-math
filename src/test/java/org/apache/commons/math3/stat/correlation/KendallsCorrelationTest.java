package org.apache.commons.math3.stat.correlation;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Assert;
import org.junit.Test;

/**
 * Test cases for Kendall's Tau rank correlation
 */
public class KendallsCorrelationTest {

    @Test
    public void testSimpleOrdered() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[i] = i;
            yArray[i] = i;
        }
        Assert.assertEquals(1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }

    @Test
    public void testSimpleReversed() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[length - i - 1] = i;
            yArray[i] = i;
        }
        Assert.assertEquals(-1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }

    @Test
    public void testSimpleOrderedPowerOf2() {
        final int length = 16;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[i] = i;
            yArray[i] = i;
        }
        Assert.assertEquals(1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }

    @Test
    public void testSimpleReversedPowerOf2() {
        final int length = 16;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[length - i - 1] = i;
            yArray[i] = i;
        }
        Assert.assertEquals(-1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }

    @Test
    public void testSimpleJumble() {
        //                                     A    B    C    D
        final double[] xArray = new double[] {1.0, 2.0, 3.0, 4.0};
        final double[] yArray = new double[] {1.0, 3.0, 2.0, 4.0};

        // 6 pairs: (A,B) (A,C) (A,D) (B,C) (B,D) (C,D)
        // (B,C) is discordant, the other 5 are concordant

        Assert.assertEquals((5 - 1) / (double) 6,
                KendallsCorrelation.correlation(xArray, yArray),
                Double.MIN_VALUE);
    }

    @Test
    public void testBalancedJumble() {
        //                                     A    B    C    D
        final double[] xArray = new double[] {1.0, 2.0, 3.0, 4.0};
        final double[] yArray = new double[] {1.0, 4.0, 3.0, 2.0};

        // 6 pairs: (A,B) (A,C) (A,D) (B,C) (B,D) (C,D)
        // (A,B) (A,C), (A,D) are concordant, the other 3 are discordant

        Assert.assertEquals(0.0,
                KendallsCorrelation.correlation(xArray, yArray),
                Double.MIN_VALUE);
    }

    @Test
    public void testOrderedTies() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[i] = i / 2;
            yArray[i] = i / 2;
        }
        // 5 pairs of points that are tied in both values.
        // 16 + 12 + 8 + 4 = 40 concordant
        // (40 - 0) / Math.sqrt((45 - 5) * (45 - 5)) = 1
        Assert.assertEquals(1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }


    @Test
    public void testAllTiesInBoth() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        Assert.assertEquals(Double.NaN, KendallsCorrelation.correlation(xArray, yArray), 0);
    }

    @Test
    public void testAllTiesInX() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            xArray[i] = i;
        }
        Assert.assertEquals(Double.NaN, KendallsCorrelation.correlation(xArray, yArray), 0);
    }

    @Test
    public void testAllTiesInY() {
        final int length = 10;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        for (int i = 0; i < length; i++) {
            yArray[i] = i;
        }
        Assert.assertEquals(Double.NaN, KendallsCorrelation.correlation(xArray, yArray), 0);
    }

    @Test
    public void testSingleElement() {
        final int length = 1;
        final double[] xArray = new double[length];
        final double[] yArray = new double[length];
        Assert.assertEquals(Double.NaN, KendallsCorrelation.correlation(xArray, yArray), 0);
    }

    @Test
    public void testTwoElements() {
        final double[] xArray = new double[] {2.0, 1.0};
        final double[] yArray = new double[] {1.0, 2.0};
        Assert.assertEquals(-1.0, KendallsCorrelation.correlation(xArray, yArray), Double.MIN_VALUE);
    }

    @Test
    public void test2dDoubleArray() {
        final double[][] input = new double[][] {
                new double[] {2.0, 1.0, 2.0},
                new double[] {1.0, 2.0, 1.0},
                new double[] {0.0, 0.0, 0.0}
        };

        final double[][] expected = new double[][] {
                new double[] {1.0, 1.0 / 3.0, 1.0},
                new double[] {1.0 / 3.0, 1.0, 1.0 / 3.0},
                new double[] {1.0, 1.0 / 3.0, 1.0}};

        Assert.assertEquals(KendallsCorrelation.computeCorrelationMatrix(input),
                new BlockRealMatrix(expected));

    }

    @Test
    public void testBlockMatrix() {
        final double[][] input = new double[][] {
                new double[] {2.0, 1.0, 2.0},
                new double[] {1.0, 2.0, 1.0},
                new double[] {0.0, 0.0, 0.0}
        };

        final double[][] expected = new double[][] {
                new double[] {1.0, 1.0 / 3.0, 1.0},
                new double[] {1.0 / 3.0, 1.0, 1.0 / 3.0},
                new double[] {1.0, 1.0 / 3.0, 1.0}};

        Assert.assertEquals(
                KendallsCorrelation.computeCorrelationMatrix(
                        new BlockRealMatrix(input)),
                new BlockRealMatrix(expected));
    }

}
