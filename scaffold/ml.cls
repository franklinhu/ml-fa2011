\ProvidesClass{ml}

\PassOptionsToClass{9pt,10pt,11pt,12pt}{article}
\LoadClassWithOptions{article}

\setlength{\parindent}{0in} % remove indentation

% math
\usepackage{amsmath}
\usepackage{amssymb}

% used to format enumerate
\usepackage{enumerate}

% used to reduce section font size
\usepackage{sectsty}
\sectionfont{\large}

% margins
\usepackage[left=1in,top=0.75in,right=1in,bottom=0.75in,nohead,nofoot]{geometry}

% tabularfw
% arg: column format
\makeatletter
\newenvironment{tabularfw}[1]{
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}#1}
}{\end{tabular*}}
\makeatother

% textnormal
% arg: text to be printed
\makeatletter
\newcommand{\+}[1]{\textnormal{#1}}
\makeatother

% alias: item <=> part
\makeatletter
\renewcommand{\part}{\item}
\makeatother

% fields
\makeatletter
\newcommand{\course}[1]{\def \@course {#1}}
\newcommand{\assignment}[1]{\def \@assignment {#1}}
\newcommand{\nameone}[1]{\def \@nameone {#1}}
\newcommand{\nametwo}[1]{\def \@nametwo {#1}}
\nameone{}
\nametwo{}
\assignment
\makeatother

% title
\makeatletter
\def \@maketitle{
  \begin{center}
    {\Large \@course}\\
    {\large \@assignment}
    
    \vspace{0.5em}
    
    \begin{tabular}{c c}
      \@nameone & \@nametwo
    \end{tabular}\\
    \small\today
  \end{center}
    
  \vspace{2em}
}
\makeatother

\makeatletter
\newenvironment{question}[1]{
  \section{#1}
  \vspace{-0.5em}
  \begin{enumerate}[(a)]
    \vspace{-0.5em}
}{\end{enumerate}}
\makeatother

