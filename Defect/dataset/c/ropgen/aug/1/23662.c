fork_exec(struct socket *so, const char *ex, int do_pty)

{

	int s;

	struct sockaddr_in addr;

	socklen_t addrlen = sizeof(addr);

	int opt;

	const char *argv[256];

	/* don't want to clobber the original */

	char *bptr;

	const char *curarg;

	int c, i, ret;

	pid_t pid;



	DEBUG_CALL("fork_exec");

	DEBUG_ARG("so = %p", so);

	DEBUG_ARG("ex = %p", ex);

	DEBUG_ARG("do_pty = %x", do_pty);



	if (do_pty == 2) {

                return 0;

	} else {

		addr.sin_family = AF_INET;

		addr.sin_port = 0;

		addr.sin_addr.s_addr = INADDR_ANY;



		if ((s = qemu_socket(AF_INET, SOCK_STREAM, 0)) < 0 ||

		    bind(s, (struct sockaddr *)&addr, addrlen) < 0 ||

		    listen(s, 1) < 0) {

			error_report("Error: inet socket: %s", strerror(errno));

			closesocket(s);



			return 0;

		}

	}



	pid = fork();

	switch(pid) {

	 case -1:

		error_report("Error: fork failed: %s", strerror(errno));

		close(s);

		return 0;



	 case 0:

                setsid();



		/* Set the DISPLAY */

                getsockname(s, (struct sockaddr *)&addr, &addrlen);

                close(s);

                /*

                 * Connect to the socket

                 * XXX If any of these fail, we're in trouble!

                 */

                s = qemu_socket(AF_INET, SOCK_STREAM, 0);

                addr.sin_addr = loopback_addr;

                do {

                    ret = connect(s, (struct sockaddr *)&addr, addrlen);

                } while (ret < 0 && errno == EINTR);



		dup2(s, 0);

		dup2(s, 1);

		dup2(s, 2);

		for (s = getdtablesize() - 1; s >= 3; s--)

		   close(s);



		i = 0;

		bptr = g_strdup(ex); /* No need to free() this */

		if (do_pty == 1) {

			/* Setup "slirp.telnetd -x" */

			argv[i++] = "slirp.telnetd";

			argv[i++] = "-x";

			argv[i++] = bptr;

		} else

		   do {

			/* Change the string into argv[] */

			curarg = bptr;

			while (*bptr != ' ' && *bptr != (char)0)

			   bptr++;

			c = *bptr;

			*bptr++ = (char)0;

			argv[i++] = g_strdup(curarg);

		   } while (c);



                argv[i] = NULL;

		execvp(argv[0], (char **)argv);



		/* Ooops, failed, let's tell the user why */

        fprintf(stderr, "Error: execvp of %s failed: %s\n",

                argv[0], strerror(errno));

		close(0); close(1); close(2); /* XXX */

		exit(1);



	 default:

		qemu_add_child_watch(pid);

                /*

                 * XXX this could block us...

                 * XXX Should set a timer here, and if accept() doesn't

                 * return after X seconds, declare it a failure

                 * The only reason this will block forever is if socket()

                 * of connect() fail in the child process

                 */

                do {

                    so->s = accept(s, (struct sockaddr *)&addr, &addrlen);

                } while (so->s < 0 && errno == EINTR);

                closesocket(s);

                socket_set_fast_reuse(so->s);

                opt = 1;

                qemu_setsockopt(so->s, SOL_SOCKET, SO_OOBINLINE, &opt, sizeof(int));

		qemu_set_nonblock(so->s);



		/* Append the telnet options now */

                if (so->so_m != NULL && do_pty == 1)  {

			sbappend(so, so->so_m);

                        so->so_m = NULL;

		}



		return 1;

	}

}
