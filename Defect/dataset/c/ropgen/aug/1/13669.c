yuv2yuvX_altivec_real(int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

		      int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

		      uint8_t *dest, uint8_t *uDest, uint8_t *vDest, int dstW, int chrDstW)

{

  const vector signed int vini = {(1 << 18), (1 << 18), (1 << 18), (1 << 18)};

  register int i, j;

  {

    int __attribute__ ((aligned (16))) val[dstW];



    for (i = 0; i < (dstW -7); i+=4) {

      vec_st(vini, i << 2, val);

    }

    for (; i < dstW; i++) {

      val[i] = (1 << 18);

    }



    for (j = 0; j < lumFilterSize; j++) {

      vector signed short l1, vLumFilter = vec_ld(j << 1, lumFilter);

      vector unsigned char perm, perm0 = vec_lvsl(j << 1, lumFilter);

      vLumFilter = vec_perm(vLumFilter, vLumFilter, perm0);

      vLumFilter = vec_splat(vLumFilter, 0); // lumFilter[j] is loaded 8 times in vLumFilter



      perm = vec_lvsl(0, lumSrc[j]);

      l1 = vec_ld(0, lumSrc[j]);



      for (i = 0; i < (dstW - 7); i+=8) {

	int offset = i << 2;

	vector signed short l2 = vec_ld((i << 1) + 16, lumSrc[j]);



	vector signed int v1 = vec_ld(offset, val);

	vector signed int v2 = vec_ld(offset + 16, val);



	vector signed short ls = vec_perm(l1, l2, perm); // lumSrc[j][i] ... lumSrc[j][i+7]



	vector signed int i1 = vec_mule(vLumFilter, ls);

	vector signed int i2 = vec_mulo(vLumFilter, ls);



	vector signed int vf1 = vec_mergeh(i1, i2);

	vector signed int vf2 = vec_mergel(i1, i2); // lumSrc[j][i] * lumFilter[j] ... lumSrc[j][i+7] * lumFilter[j]



	vector signed int vo1 = vec_add(v1, vf1);

	vector signed int vo2 = vec_add(v2, vf2);



	vec_st(vo1, offset, val);

	vec_st(vo2, offset + 16, val);



	l1 = l2;

      }

      for ( ; i < dstW; i++) {

	val[i] += lumSrc[j][i] * lumFilter[j];

      }

    }

    altivec_packIntArrayToCharArray(val,dest,dstW);

  }

  if (uDest != 0) {

    int  __attribute__ ((aligned (16))) u[chrDstW];

    int  __attribute__ ((aligned (16))) v[chrDstW];



    for (i = 0; i < (chrDstW -7); i+=4) {

      vec_st(vini, i << 2, u);

      vec_st(vini, i << 2, v);

    }

    for (; i < chrDstW; i++) {

      u[i] = (1 << 18);

      v[i] = (1 << 18);

    }



    for (j = 0; j < chrFilterSize; j++) {

      vector signed short l1, l1_V, vChrFilter = vec_ld(j << 1, chrFilter);

      vector unsigned char perm, perm0 = vec_lvsl(j << 1, chrFilter);

      vChrFilter = vec_perm(vChrFilter, vChrFilter, perm0);

      vChrFilter = vec_splat(vChrFilter, 0); // chrFilter[j] is loaded 8 times in vChrFilter



      perm = vec_lvsl(0, chrSrc[j]);

      l1 = vec_ld(0, chrSrc[j]);

      l1_V = vec_ld(2048 << 1, chrSrc[j]);



      for (i = 0; i < (chrDstW - 7); i+=8) {

	int offset = i << 2;

	vector signed short l2 = vec_ld((i << 1) + 16, chrSrc[j]);

	vector signed short l2_V = vec_ld(((i + 2048) << 1) + 16, chrSrc[j]);



	vector signed int v1 = vec_ld(offset, u);

	vector signed int v2 = vec_ld(offset + 16, u);

	vector signed int v1_V = vec_ld(offset, v);

	vector signed int v2_V = vec_ld(offset + 16, v);



	vector signed short ls = vec_perm(l1, l2, perm); // chrSrc[j][i] ... chrSrc[j][i+7]

	vector signed short ls_V = vec_perm(l1_V, l2_V, perm); // chrSrc[j][i+2048] ... chrSrc[j][i+2055]



	vector signed int i1 = vec_mule(vChrFilter, ls);

	vector signed int i2 = vec_mulo(vChrFilter, ls);

	vector signed int i1_V = vec_mule(vChrFilter, ls_V);

	vector signed int i2_V = vec_mulo(vChrFilter, ls_V);



	vector signed int vf1 = vec_mergeh(i1, i2);

	vector signed int vf2 = vec_mergel(i1, i2); // chrSrc[j][i] * chrFilter[j] ... chrSrc[j][i+7] * chrFilter[j]

	vector signed int vf1_V = vec_mergeh(i1_V, i2_V);

	vector signed int vf2_V = vec_mergel(i1_V, i2_V); // chrSrc[j][i] * chrFilter[j] ... chrSrc[j][i+7] * chrFilter[j]



	vector signed int vo1 = vec_add(v1, vf1);

	vector signed int vo2 = vec_add(v2, vf2);

	vector signed int vo1_V = vec_add(v1_V, vf1_V);

	vector signed int vo2_V = vec_add(v2_V, vf2_V);



	vec_st(vo1, offset, u);

	vec_st(vo2, offset + 16, u);

	vec_st(vo1_V, offset, v);

	vec_st(vo2_V, offset + 16, v);



	l1 = l2;

	l1_V = l2_V;

      }

      for ( ; i < chrDstW; i++) {

	u[i] += chrSrc[j][i] * chrFilter[j];

	v[i] += chrSrc[j][i + 2048] * chrFilter[j];

      }

    }

    altivec_packIntArrayToCharArray(u,uDest,chrDstW);

    altivec_packIntArrayToCharArray(v,vDest,chrDstW);

  }

}
