/* ====================================================================
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 2003-2004 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution, if
 *    any, must include the following acknowledgement:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgement may appear in the software itself,
 *    if and wherever such third-party acknowledgements normally appear.
 *
 * 4. The names "The Jakarta Project", "Commons", and "Apache Software
 *    Foundation" must not be used to endorse or promote products derived
 *    from this software without prior written permission. For written
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache"
 *    nor may "Apache" appear in their name without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */
package org.apache.commons.math.stat.univariate.rank;

import java.io.Serializable;
import java.util.Arrays;
import org.apache.commons.math.stat.univariate.AbstractUnivariateStatistic;

/**
 * @version $Revision: 1.12 $ $Date: 2004/01/29 00:49:01 $
 */
public class Percentile extends AbstractUnivariateStatistic implements Serializable {

    static final long serialVersionUID = -8091216485095130416L; 
       
    /** */
    private double percentile = 0.0;

    /**
     * Constructs a Percentile with a default percentile
     * value of 50.0.
     */
    public Percentile() {
        super();
        percentile = 50.0;
    }

    /**
     * Constructs a Percentile with the specific percentile value.
     * @param p the percentile
     */
    public Percentile(final double p) {
        this.percentile = p;
    }

    /**
     * Evaluates the double[] top the specified percentile.
     * This does not alter the interal percentile state of the
     * statistic.
     * @param values Is a double[] containing the values
     * @param p Is the percentile to evaluate to.
     * @return the result of the evaluation or Double.NaN
     * if the array is empty
     */
    public double evaluate(final double[] values, final double p) {
        return evaluate(values, 0, values.length, p);
    }

    /**
     * @see org.apache.commons.math.stat.univariate.UnivariateStatistic#evaluate(double[], int, int)
     */
    public double evaluate(
        final double[] values,
        final int start,
        final int length) {

        return evaluate(values, start, length, percentile);
    }

    /**
     * Evaluates the double[] top the specified percentile.
     * This does not alter the interal percentile state of the
     * statistic.
     * @param values Is a double[] containing the values
     * @param begin processing at this point in the array
     * @param length processing at this point in the array
     * @param p Is the percentile to evaluate to.*
     * @return the result of the evaluation or Double.NaN
     * if the array is empty
     */
    public double evaluate(
        final double[] values,
        final int begin,
        final int length,
        final double p) {

        test(values, begin, length);

        if ((p > 100) || (p <= 0)) {
            throw new IllegalArgumentException("invalid percentile value");
        }
        double n = (double) length;
        if (n == 0) {
            return Double.NaN;
        }
        if (n == 1) {
            return values[begin]; // always return single value for n = 1
        }
        double pos = p * (n + 1) / 100;
        double fpos = Math.floor(pos);
        int intPos = (int) fpos;
        double dif = pos - fpos;
        double[] sorted = new double[length];
        System.arraycopy(values, begin, sorted, 0, length);
        Arrays.sort(sorted);

        if (pos < 1) {
            return sorted[0];
        }
        if (pos >= n) {
            return sorted[length - 1];
        }
        double lower = sorted[intPos - 1];
        double upper = sorted[intPos];
        return lower + dif * (upper - lower);
    }

    /**
     * The default internal state of this percentile can be set.
     * This will return that value.
     * @return percentile
     */
    public double getPercentile() {
        return percentile;
    }

    /**
     * The default internal state of this percentile can be set.
     * This will setthat value.
     * @param p a value between 0 <= p <= 100
     */
    public void setPercentile(final double p) {
        percentile = p;
    }

}