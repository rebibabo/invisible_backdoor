static always_inline void idct(uint8_t *dst, int stride, int16_t *input, int type)

{

    int16_t *ip = input;

    uint8_t *cm = cropTbl + MAX_NEG_CROP;



    int A_, B_, C_, D_, _Ad, _Bd, _Cd, _Dd, E_, F_, G_, H_;

    int _Ed, _Gd, _Add, _Bdd, _Fd, _Hd;



    int i;



    /* Inverse DCT on the rows now */

    for (i = 0; i < 8; i++) {

        /* Check for non-zero values */

        if ( ip[0] | ip[1] | ip[2] | ip[3] | ip[4] | ip[5] | ip[6] | ip[7] ) {

            A_ = M(xC1S7, ip[1]) + M(xC7S1, ip[7]);

            B_ = M(xC7S1, ip[1]) - M(xC1S7, ip[7]);

            C_ = M(xC3S5, ip[3]) + M(xC5S3, ip[5]);

            D_ = M(xC3S5, ip[5]) - M(xC5S3, ip[3]);



            _Ad = M(xC4S4, (A_ - C_));

            _Bd = M(xC4S4, (B_ - D_));



            _Cd = A_ + C_;

            _Dd = B_ + D_;



            E_ = M(xC4S4, (ip[0] + ip[4]));

            F_ = M(xC4S4, (ip[0] - ip[4]));



            G_ = M(xC2S6, ip[2]) + M(xC6S2, ip[6]);

            H_ = M(xC6S2, ip[2]) - M(xC2S6, ip[6]);



            _Ed = E_ - G_;

            _Gd = E_ + G_;



            _Add = F_ + _Ad;

            _Bdd = _Bd - H_;



            _Fd = F_ - _Ad;

            _Hd = _Bd + H_;



            /*  Final sequence of operations over-write original inputs. */

            ip[0] = _Gd + _Cd ;

            ip[7] = _Gd - _Cd ;



            ip[1] = _Add + _Hd;

            ip[2] = _Add - _Hd;



            ip[3] = _Ed + _Dd ;

            ip[4] = _Ed - _Dd ;



            ip[5] = _Fd + _Bdd;

            ip[6] = _Fd - _Bdd;

        }



        ip += 8;            /* next row */

    }



    ip = input;



    for ( i = 0; i < 8; i++) {

        /* Check for non-zero values (bitwise or faster than ||) */

        if ( ip[1 * 8] | ip[2 * 8] | ip[3 * 8] |

             ip[4 * 8] | ip[5 * 8] | ip[6 * 8] | ip[7 * 8] ) {



            A_ = M(xC1S7, ip[1*8]) + M(xC7S1, ip[7*8]);

            B_ = M(xC7S1, ip[1*8]) - M(xC1S7, ip[7*8]);

            C_ = M(xC3S5, ip[3*8]) + M(xC5S3, ip[5*8]);

            D_ = M(xC3S5, ip[5*8]) - M(xC5S3, ip[3*8]);



            _Ad = M(xC4S4, (A_ - C_));

            _Bd = M(xC4S4, (B_ - D_));



            _Cd = A_ + C_;

            _Dd = B_ + D_;



            E_ = M(xC4S4, (ip[0*8] + ip[4*8]));

            F_ = M(xC4S4, (ip[0*8] - ip[4*8]));



            G_ = M(xC2S6, ip[2*8]) + M(xC6S2, ip[6*8]);

            H_ = M(xC6S2, ip[2*8]) - M(xC2S6, ip[6*8]);



            _Ed = E_ - G_;

            _Gd = E_ + G_;



            _Add = F_ + _Ad;

            _Bdd = _Bd - H_;



            _Fd = F_ - _Ad;

            _Hd = _Bd + H_;



            if(type==1){  //HACK

                _Gd += 16*128;

                _Add+= 16*128;

                _Ed += 16*128;

                _Fd += 16*128;

            }

            _Gd += IdctAdjustBeforeShift;

            _Add += IdctAdjustBeforeShift;

            _Ed += IdctAdjustBeforeShift;

            _Fd += IdctAdjustBeforeShift;



            /* Final sequence of operations over-write original inputs. */

            if(type==0){

                ip[0*8] = (_Gd + _Cd )  >> 4;

                ip[7*8] = (_Gd - _Cd )  >> 4;



                ip[1*8] = (_Add + _Hd ) >> 4;

                ip[2*8] = (_Add - _Hd ) >> 4;



                ip[3*8] = (_Ed + _Dd )  >> 4;

                ip[4*8] = (_Ed - _Dd )  >> 4;



                ip[5*8] = (_Fd + _Bdd ) >> 4;

                ip[6*8] = (_Fd - _Bdd ) >> 4;

            }else if(type==1){

                dst[0*stride] = cm[(_Gd + _Cd )  >> 4];

                dst[7*stride] = cm[(_Gd - _Cd )  >> 4];



                dst[1*stride] = cm[(_Add + _Hd ) >> 4];

                dst[2*stride] = cm[(_Add - _Hd ) >> 4];



                dst[3*stride] = cm[(_Ed + _Dd )  >> 4];

                dst[4*stride] = cm[(_Ed - _Dd )  >> 4];



                dst[5*stride] = cm[(_Fd + _Bdd ) >> 4];

                dst[6*stride] = cm[(_Fd - _Bdd ) >> 4];

            }else{

                dst[0*stride] = cm[dst[0*stride] + ((_Gd + _Cd )  >> 4)];

                dst[7*stride] = cm[dst[7*stride] + ((_Gd - _Cd )  >> 4)];



                dst[1*stride] = cm[dst[1*stride] + ((_Add + _Hd ) >> 4)];

                dst[2*stride] = cm[dst[2*stride] + ((_Add - _Hd ) >> 4)];



                dst[3*stride] = cm[dst[3*stride] + ((_Ed + _Dd )  >> 4)];

                dst[4*stride] = cm[dst[4*stride] + ((_Ed - _Dd )  >> 4)];



                dst[5*stride] = cm[dst[5*stride] + ((_Fd + _Bdd ) >> 4)];

                dst[6*stride] = cm[dst[6*stride] + ((_Fd - _Bdd ) >> 4)];

            }



        } else {

            if(type==0){

                ip[0*8] =

                ip[1*8] =

                ip[2*8] =

                ip[3*8] =

                ip[4*8] =

                ip[5*8] =

                ip[6*8] =

                ip[7*8] = ((xC4S4 * ip[0*8] + (IdctAdjustBeforeShift<<16))>>20);

            }else if(type==1){

                dst[0*stride]=

                dst[1*stride]=

                dst[2*stride]=

                dst[3*stride]=

                dst[4*stride]=

                dst[5*stride]=

                dst[6*stride]=

                dst[7*stride]= 128 + ((xC4S4 * ip[0*8] + (IdctAdjustBeforeShift<<16))>>20);

            }else{

                if(ip[0*8]){

                    int v= ((xC4S4 * ip[0*8] + (IdctAdjustBeforeShift<<16))>>20);

                    dst[0*stride] = cm[dst[0*stride] + v];

                    dst[1*stride] = cm[dst[1*stride] + v];

                    dst[2*stride] = cm[dst[2*stride] + v];

                    dst[3*stride] = cm[dst[3*stride] + v];

                    dst[4*stride] = cm[dst[4*stride] + v];

                    dst[5*stride] = cm[dst[5*stride] + v];

                    dst[6*stride] = cm[dst[6*stride] + v];

                    dst[7*stride] = cm[dst[7*stride] + v];

                }

            }

        }



        ip++;            /* next column */

        dst++;

    }

}
